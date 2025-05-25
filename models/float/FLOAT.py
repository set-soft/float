import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from models.wav2vec2 import Wav2VecModel
from models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from models import BaseModel
from models.float.generator import Generator
from models.float.FMT import FlowMatchingTransformer

from .helpers import print_gpu_total_free_memory, print_ram_usage


######## Main Phase 2 model ########		
class FLOAT(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(self.opt.num_prev_frames)

		# motion latent auto-encoder
		self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
		self.motion_autoencoder.requires_grad_(False)

		# condition encoders
		self.audio_encoder 		= AudioEncoder(opt)
		self.emotion_encoder	= Audio2Emotion(opt)

		# FMT; Flow Matching Transformer
		self.fmt = FlowMatchingTransformer(opt)
		
		# ODE options
		self.odeint_kwargs = {
			'atol': self.opt.ode_atol,
			'rtol': self.opt.ode_rtol,
			'method': self.opt.torchdiffeq_ode_method
		}
	
	######## Motion Encoder - Decoder ########
	@torch.no_grad()
	def encode_image_into_latent(self, x: torch.Tensor) -> list:
		x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		return x_r, x_r_lambda, x_r_feats

	@torch.no_grad()
	def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
		return r_x

# Old code to decode images
# Weaknesses: all frames stored in VRAM, all transformation done after finishing
#	  @torch.no_grad()
#	  def decode_latent_into_image(self, s_r: torch.Tensor , s_r_feats: list, r_d: torch.Tensor) -> dict:
#		  T = r_d.shape[1]
#		  d_hat = []
#		  for t in range(T):
#			  s_r_d_t = s_r + r_d[:, t]
#			  img_t, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats)
#			  d_hat.append(img_t.cpu())
#		  d_hat = torch.stack(d_hat, dim=1).squeeze()
#		  return {'d_hat': d_hat}

	# New code to decode images
	# Weaknesses: When we save the video we must apply more transformations, we also waste memory
	#             keeping images in float format instead of int pixels. Might be of interest for
	#             research.
	@torch.no_grad()
	def decode_latent_into_image(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> dict:
		T = r_d.shape[1]

		# --- Method: Pre-allocate d_hat on CPU ---
		# 1. Decode the first frame to determine the shape of a single image
		#    and to get its data type.
		s_r_d_0 = s_r + r_d[:, 0]   # This is on GPU
		img_0_gpu, _ = self.motion_autoencoder.dec(s_r_d_0, alpha=None, feats=s_r_feats)
		# img_0_gpu has shape e.g., (Batch, Channels, Height, Width)

		# 2. Pre-allocate the full d_hat tensor on CPU.
		#    The shape before squeeze will be (Batch, T, Channels, Height, Width)
		batch_size = img_0_gpu.shape[0]
		img_dims = img_0_gpu.shape[1:]  # (Channels, Height, Width)

		# Shape for the stacked tensor (before squeeze)
		# This is equivalent to the shape torch.stack(list_of_imgs, dim=1) would produce
		d_hat_shape_stacked = (batch_size, T) + img_dims

		d_hat_on_cpu = torch.empty(d_hat_shape_stacked, dtype=img_0_gpu.dtype, device='cpu')

		# 3. Place the first decoded image (moved to CPU) into the pre-allocated tensor
		d_hat_on_cpu[:, 0] = img_0_gpu.cpu()  # Slicing takes care of other dims

		# 4. Loop through the rest of the frames
		print(f"Frames: {range(T)}")
		for t in range(1, T):  # Start from 1 because frame 0 is already processed
			s_r_d_t = s_r + r_d[:, t]  # on GPU
			img_t_gpu, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats)  # on GPU
			d_hat_on_cpu[:, t] = img_t_gpu.cpu()  # Move to CPU and assign to slice
			if t % 100 == 0:
				print(f"Frame {t} {print_gpu_total_free_memory()} {print_ram_usage()}")

		print(f"Before stack {print_gpu_total_free_memory()} {print_ram_usage()}")
		# 5. Apply squeeze at the end, just like the original code
		#    This will remove dimensions of size 1. For example, if T=1,
		#    the shape (Batch, 1, C, H, W) becomes (Batch, C, H, W).
		#    Or if Batch=1, (1, T, C, H, W) becomes (T, C, H, W).
		d_hat_final = d_hat_on_cpu.squeeze()
		print(f"After stack {print_gpu_total_free_memory()} {print_ram_usage()}")

		return {'d_hat': d_hat_final}

	# New code to decode video frames
	@torch.no_grad()
	def decode_latent_into_video_frames(self, s_r: torch.Tensor, s_r_feats: list, r_d: torch.Tensor) -> torch.Tensor:
		# This function will now output a tensor suitable for write_video
		# Shape: (T, H, W, C) for a single video, or (B, T, H, W, C) if handling batch
		# Dtype: torch.uint8
		# Device: CPU

		B, T_total = r_d.shape[0], r_d.shape[1]

# All this code covers the case of 0 frames to decode, I don't think is worth keeping it
#		  if T_total == 0:
#			  # Determine expected H, W, C, or use placeholders
#			  # For this example, let's assume we can determine C,H,W by a dummy decode or from config
#			  dummy_s_r_d_0 = s_r + r_d[:, 0] if T_total > 0 else s_r # Need a valid input for shape
#			  if T_total == 0 and B > 0 and r_d.numel() == 0 : # if r_d is (B,0,Z)
#				   dummy_s_r_d_0 = s_r # s_r is (B,Z)
#			  elif T_total == 0 and r_d.numel() > 0: # r_d is (B,0,...), s_r (B,...)
#				   dummy_s_r_d_0 = s_r + r_d.sum(dim=1) # just to get a (B,Z) tensor
#
#			  # A bit hacky to get C,H,W for T=0 case without actually decoding.
#			  # Ideally, C,H,W are known from model config.
#			  # Here, we'll try a dummy decode if possible or use fixed values.
#			  try:
#				  # This dummy decode is just for shape, won't be used if T=0
#				  # We need to ensure dummy_s_r_d_0 is on the correct device and has correct Z dim
#				  if s_r.shape != dummy_s_r_d_0.shape and dummy_s_r_d_0.shape[-1] == s_r.shape[-1]: # s_r is (B,Z), r_d is (B,T,Z)
#					  # This case means r_d might be empty in time but not other dims.
#					  # If r_d is (B,0,Z), s_r+r_d[:,0] fails.
#					  # Let's assume s_r itself can be decoded, or a zero tensor of Z dim
#					  dummy_img_gpu_shape_ref, _ = self.motion_autoencoder.dec(s_r, alpha=None, feats=s_r_feats)
#				  else:
#					  dummy_img_gpu_shape_ref, _ = self.motion_autoencoder.dec(dummy_s_r_d_0, alpha=None, feats=s_r_feats)
#
#				  _, C, H, W = dummy_img_gpu_shape_ref.shape
#			  except Exception as e: # Fallback if dummy decode fails (e.g. due to T=0 logic complexity)
#				  # print(f"Warning: Could not determine C,H,W for T=0, using defaults. Error: {e}")
#				  C, H, W = 3, 64, 64 # Default/placeholder values
#
#			  # If B > 1, we output (B, 0, H, W, C)
#			  # If B = 1, we output (0, H, W, C)
#			  if B > 1:
#				  return torch.empty((B, 0, H, W, C), dtype=torch.uint8, device='cpu')
#			  else: # B == 1
#				  return torch.empty((0, H, W, C), dtype=torch.uint8, device='cpu')

		# 1. Decode the first frame to determine H, W, C and batch size (B_actual)
		s_r_d_0 = s_r + r_d[:, 0] # on GPU
		img_0_gpu, _ = self.motion_autoencoder.dec(s_r_d_0, alpha=None, feats=s_r_feats)
		# img_0_gpu has shape (B, C, H, W)

		B_actual, C, H, W = img_0_gpu.shape

		# 2. Pre-allocate the full video tensor on CPU with target shape (B, T, H, W, C) and uint8
		# This tensor will store all videos in the batch if B_actual > 1
		video_tensor_uint8_shape = (B_actual, T_total, H, W, C)
		video_tensor_final_cpu = torch.empty(video_tensor_uint8_shape, dtype=torch.uint8, device='cpu')

		# 3. Process and place the first frame
		# (B,C,H,W) -> clamp -> normalize to [0,255] -> .byte() -> permute to (B,H,W,C)
		img_0_processed_gpu = ((img_0_gpu.clamp(-1, 1) + 1) / 2 * 255).byte()
		video_tensor_final_cpu[:, 0] = img_0_processed_gpu.permute(0, 2, 3, 1).cpu() # (B,C,H,W) -> (B,H,W,C)

		# 4. Loop through the rest of the frames
		print(f"Frames: {range(T_total)}")
		for t in range(1, T_total):
			s_r_d_t = s_r + r_d[:, t] # on GPU
			img_t_gpu, _ = self.motion_autoencoder.dec(s_r_d_t, alpha=None, feats=s_r_feats) # (B,C,H,W) on GPU

			# Process: clamp, normalize, convert to byte
			img_t_processed_gpu = ((img_t_gpu.clamp(-1, 1) + 1) / 2 * 255).byte() # (B,C,H,W), uint8, on GPU

			# Permute to (B,H,W,C) and move to CPU slice
			video_tensor_final_cpu[:, t] = img_t_processed_gpu.permute(0, 2, 3, 1).cpu()

			if t % 100 == 0:
				print(f"Frame {t} {print_gpu_total_free_memory()} {print_ram_usage()}")

		print(f"Before stack {print_gpu_total_free_memory()} {print_ram_usage()}")
		# If the original batch size B was 1, we might want to squeeze the batch dimension
		# to match the (T,H,W,C) expected by torchvision.io.write_video for a single video.
		# The original `save_video` implied it handled a single video path,
		# so if B > 1, it would likely process video_tensor_final_cpu[0].
		if B_actual == 1:
			return video_tensor_final_cpu.squeeze(0) # Shape: (T, H, W, C)
		else:
			return video_tensor_final_cpu # Shape: (B, T, H, W, C)

	######## Motion Sampling and Inference ########
	@torch.no_grad()
	def sample(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		r_cfg_scale: float = 1.0,
		e_cfg_scale: float = 1.0,
		emo: str = None,
		nfe: int = 10,
		seed: int = None
	) -> torch.Tensor:

		r_s, a = data['r_s'], data['a']
		B = a.shape[0]

		# make time 
		time = torch.linspace(0, 1, self.opt.nfe, device=self.opt.rank)
		
		# encoding audio first with whole audio
		a = a.to(self.opt.rank)
		T = math.ceil(a.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		wa = self.audio_encoder.inference(a, seq_len=T)

		# encoding emotion first
		emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
		if emo_idx is None:
			we = self.emotion_encoder.predict_emotion(a).unsqueeze(1)
		else:
			we = F.one_hot(torch.tensor(emo_idx, device = a.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0)

		sample = []
		# sampleing chunk by chunk
		for t in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				seed = self.opt.seed if seed is None else seed	
				g = torch.Generator(self.opt.rank)
				g.manual_seed(seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

			if t == 0: # should define the previous
				prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_wa_t = wa_t[:, -self.num_prev_frames:]
			
			wa_t = wa[:, t * self.num_frames_for_clip: (t+1)*self.num_frames_for_clip]

			if wa_t.shape[1] < self.num_frames_for_clip: # padding by replicate
				wa_t = F.pad(wa_t, (0, 0, 0, self.num_frames_for_clip - wa_t.shape[1]), mode='replicate')

			def sample_chunk(tt, zt):
				out = self.fmt.forward_with_cfv(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						wa 			= wa_t, 			 
						wr 			= r_s,
						we 			= we, 
						prev_x 		= prev_x_t, 	
						prev_wa 	= prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale
						)

				out_current = out[:, self.num_prev_frames:]
				return out_current

			# solve ODE
			trajectory_t = odeint(sample_chunk, x0, time, **self.odeint_kwargs)
			sample_t = trajectory_t[-1]
			sample.append(sample_t)
		sample = torch.cat(sample, dim=1)[:, :T]
		return sample

	@torch.no_grad()
	def inference(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
		nfe			= 10,
		seed		= None,
		ret_d_hat	= True,  # True if we want the images (float), False for video frames
	) -> dict:

		s, a = data['s'], data['a']
		s_r, r_s_lambda, s_r_feats = self.encode_image_into_latent(s.to(self.opt.rank))
		if 's_r' in data:
			r_s = self.encode_identity_into_motion(s_r)
		else:
			r_s = self.motion_autoencoder.dec.direction(r_s_lambda)
		data['r_s'] = r_s

		# set conditions
		if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
		if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
		if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale

		sample = self.sample(data, a_cfg_scale = a_cfg_scale, r_cfg_scale = r_cfg_scale, e_cfg_scale = e_cfg_scale, emo = emo, nfe = nfe, seed = seed)
		if ret_d_hat:
			data_out = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
		else:
			data_out = self.decode_latent_into_video_frames(s_r = s_r, s_r_feats = s_r_feats, r_d = sample)
		return data_out




################ Condition Encoders ################
class AudioEncoder(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.only_last_features = opt.only_last_features
		
		self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(opt.num_prev_frames)

		self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
		self.wav2vec2.feature_extractor._freeze_parameters()

		for name, param in self.wav2vec2.named_parameters():
			param.requires_grad = False

		audio_input_dim = 768 if opt.only_last_features else 12 * 768

		self.audio_projection = nn.Sequential(
			nn.Linear(audio_input_dim, opt.dim_w),
			nn.LayerNorm(opt.dim_w),
			nn.SiLU()
			)

	def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
		if self.only_last_features:
			a = a.last_hidden_state
		else:
			a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
			a = a.reshape(a.shape[0], a.shape[1], -1)
		return a

	def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
		if prev_a is not None:
			a = torch.cat([prev_a, a], dim = 1)
			if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
		else:
			if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
	
		return self.audio_projection(a) # frame by frame

	@torch.no_grad()
	def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
			a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
		a = self.get_wav2vec2_feature(a, seq_len=seq_len)
		return self.audio_projection(a)



class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(opt.audio2emotion_path, local_files_only=True)
        self.wav2vec2_for_emotion.eval()
        
		# seven labels
        self.id2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
						4: "neutral", 5: "sad", 6: "surprise"}

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
        logits = self.wav2vec2_for_emotion.forward(a).logits
        return F.softmax(logits, dim=1) 	# scores

#######################################################