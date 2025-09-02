

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colormaps as cm
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

class Tensor:
	def __init__(self, tensor_dims):
		self.dims = tensor_dims
		self.order = len(tensor_dims)
		self.rows = tensor_dims[0]
		self.cols = tensor_dims[1]
		self.slices = tensor_dims[2] if self.order > 2 else 1
		self.blocks = tensor_dims[3] if self.order > 3 else 1

class TensorVisualization:
	def __init__(self, tensor):
		self.tensor = tensor
		self.t_dims = tensor.dims
		self.t_order = tensor.order
	
	def draw_tensor(self, highlight_fiber=None, matricize_mode=None):
		t_dims = self.t_dims
		mxzn_modes = ["matricize_mode_1", "matricize_mode_2", "matricize_mode_3"]
		if matricize_mode and highlight_fiber and highlight_fiber["mode"] != matricize_mode:
			print(f"Using matricization mode {highlight_fiber['mode']} as specified in highlight_fiber argument.")
			return
		if highlight_fiber:
			matricize_mode = highlight_fiber["mode"]

		if self.t_order == 1:
			self._draw_order1_tensor(t_dims[0])
		elif self.t_order == 2:
			self._draw_order2_tensor(t_dims[0], t_dims[1])
		elif self.t_order == 3:
			print(f"t_order=3? {self.t_order}")
			color_map = self._draw_order3_tensor(t_dims[0], t_dims[1], t_dims[2], highlight_fiber)
		elif self.t_order == 4:
			print(f"t_order=4? {self.t_order}")
			color_map = self._draw_order4_tensor(t_dims[0], t_dims[1], t_dims[2], t_dims[3], highlight_fiber, show=False)
		
		if matricize_mode:
			mm = matricize_mode - 1
			if self.tensor.order == 4:
				views = [(mxzn_modes[mm], "Mode-1 Matricization", self._draw_mode1_4d_matricization),
						(mxzn_modes[mm], "Mode-2 Matricization", self._draw_mode2_4d_matricization),
						(mxzn_modes[mm], "Mode-3 Matricization", self._draw_mode3_4d_matricization)]
			else:
				print("yes matzn mode, not order 4")
				views = [(mxzn_modes[mm], "Mode-1 Matricization", self._draw_mode1_matricization),
						(mxzn_modes[mm], "Mode-2 Matricization", self._draw_mode2_matricization),
						(mxzn_modes[mm], "Mode-3 Matricization", self._draw_mode3_matricization)]

			# print(f"views={list(views)}")
			for i in range(len(mxzn_modes)):
				if i + 1 == matricize_mode:
					views[i] = (True, views[i][1], views[i][2])
				else:
					views[i] = (False, views[i][1], views[i][2])
			active_views = [v for v in views if v[0]]
			print(f"active_views={list(active_views)}")
			n_views = len(active_views)
			if n_views > 0:
				fig = plt.figure(figsize=(5,  5 * n_views))
				for i, (_, title, func) in enumerate(active_views):
					ax = fig.add_subplot(len(active_views), 1, i + 1, projection='3d')
					# ^^ add_subplot syntax: rows, cols, index, kwargs* 
					func(t_dims, color_map, highlight_fiber, ax=ax, title=title)
				plt.show()

	def lighten_color(self, color, row_idx=0, depth_idx=0, row_weight=0.15, depth_weight=0.0):
		if isinstance(color, tuple) and len(color) == 4:
			color = color[:3]
		color = mcolors.to_rgb(color)
		factor = 1 - (row_weight * row_idx + depth_weight * depth_idx)
		# print(f"factor={factor}, row_idx={row_idx}, depth_idx={depth_idx}")
		factor = 1 - (row_weight * row_idx + depth_weight * depth_idx)
		# print(f"factor={factor}, row_idx={row_idx}, depth_idx={depth_idx}")
		factor = max(0.0, min(1.0, factor))
		return [(1.0 - (1.0 - c) * factor) for c in color]

	def make_cube_faces(self, x, y, z, size=1):
		return [
			[(x, y, z), (x+size, y, z), (x+size, y+size, z), (x, y+size, z)],
			[(x, y, z+size), (x+size, y, z+size), (x+size, y+size, z+size), (x, y+size, z+size)],
			[(x, y, z), (x, y+size, z), (x, y+size, z+size), (x, y, z+size)],
			[(x+size, y, z), (x+size, y+size, z), (x+size, y+size, z+size), (x+size, y, z+size)],
			[(x, y, z), (x+size, y, z), (x+size, y, z+size), (x, y, z+size)],
			[(x, y+size, z), (x+size, y+size, z), (x+size, y+size, z+size), (x, y+size, z+size)]
			]


	def _draw_order1_tensor(self):
		n = self.t_dims[0]
		fig, ax = plt.subplots()
		cmap = plt.get_cmap('hsv', n)
		gap = 0.25
		for i in range(n):
			color = self.lighten_color(cmap(i), row_idx=i)
			ax.add_patch(Rectangle((i * (1 + gap), 0), 1, 1, color=color, ec='black'))
		ax.set_xlim(-0.5, n * (1 + gap) - gap + 0.5)
		ax.set_ylim(-0.5, 1.5)
		ax.set_aspect('equal')
		ax.axis('off')
		plt.title(f'1st-Order Tensor: Shape ({n},)')
		plt.show()

	def _draw_order2_tensor(self, n_rows, n_cols):
		fig, ax = plt.subplots()
		gap = 0.25
		cmap = plt.get_cmap('hsv', n_cols)
		for i in range(n_rows):
			for j in range(n_cols):
				x = (n_rows - 1 - i) * (1 + gap)
				y = j * (1 + gap)
				color = self.lighten_color(cmap(j), row_idx=i)
				ax.add_patch(Rectangle((x, y), 1, 1, color=color, ec='black'))
		ax.set_xlim(-0.5, n_rows * (1 + gap) - gap + 0.5)
		ax.set_ylim(-0.5, n_cols * (1 + gap) - gap + 0.5)
		ax.set_aspect('equal')
		ax.axis('off')
		plt.title(f'2nd-Order Tensor: Shape ({n_rows}, {n_cols})')
		plt.show()


	def create_label(self, xyz, ijk):
		lab = (r'$ %s_%s $' %(xyz, ijk))
		return lab

	def _draw_order3_tensor(self, n_rows, n_cols, depth, highlight_fiber=None):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		gap = 0.25
		color_map = {}
		cmap = plt.get_cmap('jet', n_cols * depth)
		for idx, (j, k) in enumerate([(j, k) for k in range(depth) for j in range(n_cols)]):
			base_color = cmap(idx)
			for i in range(n_rows):
				color = self.lighten_color(base_color, row_idx=i, depth_idx=k)
				color_map[(i, j, k)] = color

		highlight_set = set()
		if highlight_fiber:
			mode, idx = highlight_fiber["mode"], highlight_fiber["index"]
			if mode == 1:
				highlight_set = {(i, idx[0], idx[1]) for i in range(n_rows)}
			elif mode == 2:
				highlight_set = {(idx[0], j, idx[1]) for j in range(n_cols)}
			elif mode == 3:
				highlight_set = {(idx[0], idx[1], k) for k in range(depth)}

		for (i, j, k), color in color_map.items():
			x = j * (1 + gap)
			y = k * (1 + gap)
			z = (n_rows - 1 - i) * (1 + gap)
			lw = 1.5 if (i, j, k) in highlight_set else 0.3
			faces = self.make_cube_faces(x, y, z)
			ax.add_collection3d(Poly3DCollection(
				faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))

		ax.set_xlim(-2, n_cols * (1 + gap))
		ax.set_ylim(-1, depth * (1 + gap))
		ax.set_zlim(0, n_rows * (1 + gap) + 1)
		ax.set_box_aspect((n_cols, depth, n_rows))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		plt.title(f'3rd-Order Tensor: Shape ({n_rows}, {n_cols}, {depth})')

		# --- Add axis-aligned labels and arrows ---
		arrow_len = 0.75
		fontsize = 10

		# Y-axis labels (strata)
		for i in range(n_rows):
			quiver_yaxis_z0 = qyz0 = ((i + 1) * 1) + (i * gap) - 0.5
			quiver_yaxis_z1 = qyz1 = 0
			quiver_yaxis_y0 = qyy0 = 0
			quiver_yaxis_y1 = qyy1 = 0
			quiver_yaxis_x0 = qyx0 = -1.25 # express ITO params 
			quiver_yaxis_x1 = qyx1 = +1.00 # express ITO params 
			text_yaxis_x = tyx = -1.20
			text_yaxis_y = tyy = -1.00
			text_yaxis_z = tyz = qyz0 + 0.5 # for slant 

			z0 = (n_rows - 1 - i) * (1 + gap) + 0.5
			ax.text(tyx, tyy, tyz, self.create_label('y', i), fontsize=fontsize, color='black')
			ax.quiver(qyx0, qyy0, qyz0, \
					  qyx1, qyy1, qyz1, arrow_length_ratio=0.2, color='black')

		# X-axis labels (slices)
		for j in range(n_cols):
			x0 = j * (1 + gap) + 0.5
			y0 = depth * (1 + gap) + 0.5
			z0 = n_rows * (1 + gap)
			# print(f"xyz0 x-axis: {x0}, {y0}, {z0}")
			# print(f"arrow_len = {arrow_len}")
			# xyz0 x-axis: 0.5, 3.0, 5.0
			# xyz0 x-axis: 1.75, 3.0, 5.0
			# xyz0 x-axis: 3.0, 3.0, 5.0
			# xyz0 x-axis: 4.25, 3.0, 5.0
			# for 4,4,2
			ax.text(x0 - 0.2, y0 + 0.7, z0 + 0.3, self.create_label('x', j), fontsize=fontsize)
			ax.quiver(x0 - 0.2, y0 + 0.75, z0 + 0.2, \
					0, -arrow_len, -arrow_len * 0.5, arrow_length_ratio=0.2, color='black')

		# Z-axis labels (panels)
		for k in range(depth):
			x0 = 0.25
			y0 = k * (1 + gap) + 0.5
			z0 = n_rows * (1 + gap) + 0.5
			print(f"xyz0 z-axis: {x0}, {y0}, {z0}")
			# xyz0 z-axis: 0.25, 0.5, 5.5
			# xyz0 z-axis: 0.25, 1.75, 5.5
			# for 4,4,2
			ax.text(x0 - 1.25, y0-0.5, z0, self.create_label('z', k), fontsize=fontsize)
			ax.quiver(x0 - 1.0, y0, z0-0.2, \
			 		  0.5, 0, -0.5, arrow_length_ratio=0.2, color='black')

		plt.show()
		return color_map


	def _draw_order4_tensor(self, n_rows, n_cols, depth, n_blocks, highlight_fiber=None, show=True):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		gap = 0.25
		block_gap = 1.75
		cmap = plt.get_cmap('jet', n_cols * depth * n_blocks)
		color_map = {}
		idx = 0
		for l in range(n_blocks):
			for k in range(depth):
				for j in range(n_cols):
					base_color = cmap(idx)
					for i in range(n_rows):
						color = self.lighten_color(base_color, row_idx=i, depth_idx=k)
						color_map[(i, j, k, l)] = color
					idx += 1
		highlight_set = set()
		if highlight_fiber:
			mode, idx = highlight_fiber["mode"], highlight_fiber["index"]
			try:
				if mode == 1:
					j, k, l = idx
					highlight_set = {(i, j, k, l) for i in range(n_rows)}
				elif mode == 2:
					i, k, l = idx
					highlight_set = {(i, j, k, l) for j in range(n_cols)}
				elif mode == 3:
					i, j, l = idx
					highlight_set = {(i, j, k, l) for k in range(depth)}
				elif mode == 4:
					i, j, k = idx
					highlight_set = {(i, j, k, l) for l in range(n_blocks)}
			except ValueError:
				raise ValueError(f"Mode-{mode} highlighting expects an index with 3 elements, but got: {idx}")
		
		# cube creation (each iteration of faces is one cube (list of length 6))
		for (i, j, k, l), color in color_map.items():
			x = j * (1 + gap)
			y = k * (1 + gap)
			z = (n_rows - 1 - i) * (1 + gap) + l * (depth + 1 + block_gap)
			lw = 1.5 if (i, j, k, l) in highlight_set else 0.3
			faces = self.make_cube_faces(x, y, z)
			if l == 0:
				print(f"l=0: faces type and len = {type(faces)}, {len(faces)}")
				ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='magenta', linewidths=lw, alpha=0.4))
			else:
				ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
		
		ax.set_xlim(0, n_cols * (1 + gap))
		ax.set_ylim(0, n_rows * n_blocks * (1 + gap))
		ax.set_zlim(0, (n_rows + block_gap + 0.75) * n_blocks)
		ax.set_box_aspect((n_cols, depth * n_rows, n_rows * n_blocks + (1 + block_gap)))
		ax.axis('off')
		ax.view_init(elev=15, azim=300)
		plt.title(f'4th-Order Tensor: Shape ({n_rows}, {n_cols}, {depth}, {n_blocks})')

		# --- Add axis-aligned labels and arrows ---






		if show:
			# plt.title(f'4th-Order Tensor: Shape ({n_rows}, {n_cols}, {depth}, {n_blocks})')
			plt.show()
		return color_map


	def _draw_mode1_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth = t_dims
		gap = 0.25
		matrix_cols = n_cols * depth
		mode1_highlight = None
		if highlight_fiber and highlight_fiber["mode"] == 1:
			mode1_highlight = highlight_fiber["index"]

		col = 0
		for k in range(depth):
			for j in range(n_cols):
				for i in range(n_rows):
					x = col * (1 + gap)
					y = 0
					z = (n_rows - 1 - i) * (1 + gap)
					color = color_map[(i, j, k)]
					lw = 1.5 if mode1_highlight == (j, k) else 0.3
					faces = self.make_cube_faces(x, y, z)
					ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
				col += 1

		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, n_rows * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, n_rows))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

	def _draw_mode2_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth = t_dims
		gap = 0.25
		matrix_cols = n_rows * depth
		mode2_highlight = None
		if highlight_fiber and highlight_fiber["mode"] == 2:
			mode2_highlight = highlight_fiber["index"]

		col = 0
		for k in range(depth):
			for i in range(n_rows):
				for j in range(n_cols):
					x = col * (1 + gap)
					y = 0
					z = (n_cols - 1 - j) * (1 + gap)
					color = color_map[(i, j, k)]
					lw = 1.5 if mode2_highlight == (i, k) else 0.3
					faces = self.make_cube_faces(x, y, z)
					ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
				col += 1

		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, n_cols * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, n_cols))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

	def _draw_mode3_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth = t_dims
		gap = 0.25
		matrix_cols = n_rows * n_cols
		mode3_highlight = None
		if highlight_fiber and highlight_fiber["mode"] == 3:
			mode3_highlight = highlight_fiber["index"]

		col = 0
		for j in range(n_cols):
			for i in range(n_rows):
				for k in range(depth):
					x = col * (1 + gap)
					y = 0
					z = (depth - 1 - k) * (1 + gap)
					color = color_map[(i, j, k)]
					lw = 1.5 if mode3_highlight == (i, j) else 0.3
					faces = self.make_cube_faces(x, y, z)
					ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
				col += 1

		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, depth * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, depth))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

	def _draw_mode1_4d_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth, blocks, = t_dims
		gap = 0.25
		matrix_cols = n_cols * depth * blocks
		col = 0
		highlight = highlight_fiber["index"] if highlight_fiber and highlight_fiber["mode"] == 1 else None
		for l in range(blocks):
			for k in range(depth):
				for j in range(n_cols):
					for i in range(n_rows):
						x = col * (1 + gap)
						y = 0
						z = (n_rows - 1 - i) * (1 + gap)
						color = color_map[(i, j, k, l)]
						lw = 1.5 if highlight == (j, k, l) else 0.3
						faces = self.make_cube_faces(x, y, z)
						ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
					col += 1
		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, n_rows * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, n_rows))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

	def _draw_mode2_4d_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth, blocks, = t_dims
		gap = 0.25
		matrix_cols = n_rows * depth * blocks
		col = 0
		highlight = highlight_fiber["index"] if highlight_fiber and highlight_fiber["mode"] == 2 else None
		for l in range(blocks):
			for k in range(depth):
				for i in range(n_rows):
					for j in range(n_cols):
						x = col * (1 + gap)
						y = 0
						z = (n_cols - 1 - j) * (1 + gap)
						color = color_map[(i, j, k, l)]
						lw = 1.5 if highlight == (i, k, l) else 0.3
						faces = self.make_cube_faces(x, y, z)
						ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
					col += 1
		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, n_cols * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, n_cols))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

	def _draw_mode3_4d_matricization(self, t_dims, color_map, highlight_fiber=None, ax=None, title=""):
		n_rows, n_cols, depth, blocks, = t_dims
		gap = 0.25
		matrix_cols = n_rows * n_cols * blocks
		col = 0
		highlight = highlight_fiber["index"] if highlight_fiber and highlight_fiber["mode"] == 3 else None
		for l in range(blocks):
			for j in range(n_cols):
				for i in range(n_rows):
					for k in range(depth):
						x = col * (1 + gap)
						y = 0
						z = (depth - 1 - k) * (1 + gap)
						color = color_map[(i, j, k, l)]
						lw = 1.5 if highlight == (i, j, l) else 0.3
						faces = self.make_cube_faces(x, y, z)
						ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='black', linewidths=lw, alpha=0.4))
					col += 1
		ax.set_xlim(0, matrix_cols * (1 + gap))
		ax.set_ylim(0, 1 + gap)
		ax.set_zlim(0, depth * (1 + gap))
		ax.set_box_aspect((matrix_cols, 1, depth))
		ax.axis('off')
		ax.view_init(elev=30, azim=300)
		ax.set_title(title)

