from tinygrad import Tensor

class TransformerBlock:
	def __init__(self, embed_dim, num_heads, feed_forward_dim, prenorm=False, activation=lambda x: x.relu(), dropout=0.1):
		assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

		self.num_heads = num_heads
		self.head_size = embed_dim // num_heads
		self.prenorm, self.activation = prenorm, activation
		self.dropout = dropout

		self.query = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
		self.key = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))
		self.value = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

		self.output = (Tensor.scaled_uniform(embed_dim, embed_dim), Tensor.zeros(embed_dim))

		self.feed_forward1 = (Tensor.scaled_uniform(embed_dim, feed_forward_dim), Tensor.zeros(feed_forward_dim))
		self.feed_forward2 = (Tensor.scaled_uniform(feed_forward_dim, embed_dim), Tensor.zeros(embed_dim))

		self.layer_norm1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
		self.layer_norm2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

	def attention(self, x):
		# x: (batch_size, time, embed_dim) -> (batch_size, time, embed_dim)
		query, key, value = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.query, self.key, self.value]]
		attention = Tensor.scaled_dot_product_attention(query, key, value).transpose(1,2)
		return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.output)

	def __call__(self, x):
		if self.prenorm:
			x = x + self.attention(x.layernorm().linear(*self.layer_norm1)).dropout(self.dropout)
			x = x + self.activation(x.layernorm().linear(*self.layer_norm2).linear(*self.feed_forward1)).linear(*self.feed_forward2).dropout(self.dropout)
		else:
			x = x + self.attention(x).dropout(self.dropout)
			x = x.layernorm().linear(*self.layer_norm1)
			x = x + self.activation(x.linear(*self.feed_forward1)).linear(*self.feed_forward2).dropout(self.dropout)
			x = x.layernorm().linear(*self.layer_norm2)
		return x

class Transformer:
	def __init__(self, symbols, max_length, layers, embed_dim, num_heads, feed_forward_dim):
		self.max_length, self.symbols = max_length, symbols
		self.embedding = Tensor.scaled_uniform(max_length + symbols, embed_dim, requires_grad=False)
		self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, feed_forward_dim) for _ in range(layers)]
		self.final_layer = Tensor.scaled_uniform(embed_dim, symbols)

	def forward(self, x):
		batch_size = x.shape[0]

		max_length_eye = Tensor.eye(x.shape[1])
		max_length_eye = max_length_eye.unsqueeze(0).expand([batch_size, *max_length_eye.shape])

		onehot_feature = x.one_hot(self.symbols)

		onehot = max_length_eye.cat(onehot_feature, dim=2).flatten(end_dim=1)

		x = onehot.dot(self.embedding).reshape((batch_size, x.shape[1], -1))
		x = x.sequential(self.transformer_blocks)
		x = x.reshape((-1, x.shape[-1])).dot(self.final_layer).log_softmax()
		return x.reshape((batch_size, -1, x.shape[-1]))