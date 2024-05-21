from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

# 将 GloVe 文件转换为 word2vec 格式
# glove_input_file = 'glove.840B.300d.txt'
word2vec_output_file = 'glove.840B.300d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)

# 加载转换后的文件
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# 保存为 gensim 格式
model.save('glove.840B.300d.gensim')