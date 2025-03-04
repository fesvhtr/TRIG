import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans as kmeans
from .improved_precision_recall import compute_pairwise_distances



class MANIFOLD(object):
	@torch.no_grad()
	def __init__(self, real_features, fake_features, metric = 'euclidian', device = 'cpu'):

		self.metric = metric
		self.device = device

		# 确保输入是PyTorch张量
		if isinstance(real_features, np.ndarray):
			real_features = torch.from_numpy(real_features)
		if isinstance(fake_features, np.ndarray):
			fake_features = torch.from_numpy(fake_features)
			
		self.real_features = real_features.to(self.device)
		self.fake_features = fake_features.to(self.device)

		self.num_reals = real_features.shape[0]
		self.num_fakes = fake_features.shape[0]

		print('计算成对距离...')
		self.real2real_distances = compute_pairwise_distances(self.real_features, metric = self.metric, device = self.device)

		# self.real2real_sorted = np.sort(self.real2real_distances, axis = 1)
		# self.real2real_sorted_ids = self.real2real_distances.argsort(axis = 1)
		self.real2real_sorted = self.real2real_distances.sort(dim = 1)[0].detach().cpu().numpy()
		self.real2real_sorted_ids = self.real2real_distances.argsort(dim = 1).detach().cpu().numpy()

		torch.cuda.empty_cache()


		# for clustering
		self.num_cluster = 0
		self.modes = None
		self.sample_mode_ids = None


		print('初始化完成！')

	def is_in_ball(self, k = 3, samples = None, cluster = False):
		""" Compute the differences between radii of kNN balls and distances
			for judging whether they are in each kNN ball or not.

			args:
				k (int): real ball's size is distance between reference real sample 
						 and k th nearest real sample.
				samples (np.array, num_samples * embed_dim): embedded generation samples
			return:
				dist_radi (np.array, num_reals * num_samples): each element means 
						 (radii of a ball - (distance between center of the ball and a sample)).
						 if it is larger than 0, it is out of the ball.
				r (np.array, num_reals * 1): radii of each kNN real balls
				out_ball_ids (np.array, num_out_ball_samples): indices of samples outside of balls.
		"""
		if samples is not None:
			if isinstance(samples, np.ndarray):
				samples = torch.from_numpy(samples)
			samples = samples.to(self.device)
			
		if not cluster:
			real2samples_distances = compute_pairwise_distances(self.real_features, samples, metric = self.metric, device = self.device)
		else:
			real2samples_distances = compute_pairwise_distances(self.modes, samples, metric = self.metric, device = self.device)
		real2samples_distances = real2samples_distances.detach().cpu().numpy()

		r = self.real2real_sorted[:,k] if not cluster else self.mode2mode_sorted[:,k]

		dist_radi = (r[:,None].repeat(samples.shape[0], axis = 1) - real2samples_distances)
		out_ball_ids = np.where((dist_radi > 0).any(axis = 0) == False)[0]

		return dist_radi, r, out_ball_ids

	def rarity(self, k = 3, samples = None):
		""" 计算稀有度分数
		
		Args:
			k (int): k近邻的k值，如果大于样本数量会自动调整
			samples (np.array, N * embed_dim): 嵌入空间中的生成样本
		Returns:
			scores (np.array, num_samples): 每个样本的分数（未排序）
			scores_ids (np.array, num_samples_in_valid_ball): 有效球内样本的排序索引（降序）
		"""
		# 确保k不超过样本数量
		k = min(k, self.num_reals - 1)
		if k < 1:
			raise ValueError(f"k值({k})必须大于0且小于样本数量({self.num_reals})")

		samples = self.fake_features if samples is None else samples
		in_ball_dist, r, out_ball_ids = self.is_in_ball(samples = samples, k = k)

		num_out_ball = len(out_ball_ids)
		valid_real_balls = (in_ball_dist>0)

		scores = np.zeros(samples.shape[0])
		
		print('计算每个样本的稀有度分数...')
		for i in tqdm(range(samples.shape[0]), desc="计算稀有度", ncols=100):
			if i not in out_ball_ids:
				scores[i] = r[valid_real_balls[:,i]].min() if valid_real_balls[:,i].any() else 0

		scores_ids = (-scores).argsort()[:samples.shape[0] - num_out_ball]

		return scores, scores_ids


























