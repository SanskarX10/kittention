import numpy as np
from sentence_transformers import SentenceTransformer



np.random.seed(42)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed(text):
    emb_vec = np.array([model.encode(token) for token in text.split(" ")])
    return emb_vec                                    # (num_words , embed dim)

def seq_len(text):
    return len(text.split(" "))

def softmax(vec):
    vec = vec - np.max(vec, axis=-1, keepdims=True)
    vec = np.exp(vec)
    sum_exp = np.sum(vec, axis=-1, keepdims=True)
    return vec / sum_exp



class Attn:
    def __init__(self, x=None, d_k=None):
        self.x = x
        self.d_k = d_k

    def selfattn(self,x):

        cxt_len = seq_len(x)
        emb_vec = embed(x)                            # input -> embd_vec

        print("cxt_len: ", cxt_len)
        print("embed vec len: ", np.shape(emb_vec))

        W_Q = np.random.rand(len(emb_vec[0]), self.d_k)  # embed_dim, dk
        W_K = np.random.rand(len(emb_vec[0]), self.d_k)
        W_V = np.random.rand(len(emb_vec[0]), self.d_k)

        print("weight mat shape: ", np.shape(W_Q))

        Q = np.matmul(emb_vec, W_Q)     # num_words, embed_dim x embed dim, dk
        K = np.matmul(emb_vec, W_K)
        V = np.matmul(emb_vec, W_V)

        print("shape of query vec: ", np.shape(Q))
        scores = np.dot(Q, np.transpose(K)) # num_words , num_words
        scores /= np.sqrt(self.d_k) 
        scores = softmax(scores)
        scores = np.dot(scores, V)
        print("shape of scores mat: ", np.shape(scores))
        return scores
    
    def multiheadattn(self, x, n_heads):

        cxt_len = seq_len(x)
        emb_vec = embed(x)                            # input -> embd_vec

        print("cxt_len: ", cxt_len)
        print("embed vec len: ", np.shape(emb_vec))

        W_Q = np.random.rand(len(emb_vec[0]), n_heads * self.d_k) # embed_dim , n_heads * dk
        W_K = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_V = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_O = np.random.rand(n_heads * self.d_k, len(emb_vec[0]))


        print("weight mat shape: ", np.shape(W_Q))

        Q = np.matmul(emb_vec, W_Q).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)   # num_words, embed_dim x embed dim, n_heads * dk -> num_words, n_heads * dk
        K = np.matmul(emb_vec, W_K).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)
        V = np.matmul(emb_vec, W_V).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)
        print("shape of query vec: ", np.shape(Q))    # n_heads, num_words, dk

        all_heads = []
        for h in range(n_heads):
            head_score = np.dot(Q[h], np.transpose(K[h]))
            head_score /= np.sqrt(self.d_k)
            head_score = softmax(head_score)
            head_score = np.matmul(head_score, V[h])
            all_heads.append(head_score)

        scores = np.concatenate(all_heads, axis=-1)
        scores = np.matmul(scores, W_O)
        print("shape of scores: ", np.shape(scores))
        return scores

    def sparseattn(self, x, n_heads, sparsity, stride=2, local_window = 2):
        """ https://arxiv.org/pdf/1904.10509 """
        
        cxt_len = seq_len(x)
        emb_vec = embed(x)  # input -> embd_vec
        mask = np.zeros((cxt_len, cxt_len))

        print("cxt_len: ", cxt_len)
        print("embed vec len: ", np.shape(emb_vec))

        W_Q = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)  # embed_dim , n_heads * dk
        W_K = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_V = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_O = np.random.rand(n_heads * self.d_k, len(emb_vec[0]))


        Q = np.matmul(emb_vec, W_Q).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)   # num_words, embed_dim x embed dim, n_heads * dk -> num_words, n_heads * dk
        K = np.matmul(emb_vec, W_K).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)
        V = np.matmul(emb_vec, W_V).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)
        print("shape of query vec: ", np.shape(Q))    # n_heads, num_words, dk

        if sparsity == "strided":
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if j <= i and (i-j) % stride == 0:
                        mask[i, j] = 0
                    else:
                        mask[i,j] = -1e9
        
        if sparsity == "fixed":
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if j <= i and ((j >= i - local_window) or (j % stride == 0)):
                        cond1, cond2, cond3 =str(j <= i), str(j >= i - local_window), str(j % stride == 0)
                        print(f"given i = {i}, j = {j} ; and j < = i {cond1} with j > = i - local_window {cond2} or j % stride is {cond3}") # debug statement
                        mask[i, j] = 0
                    else:
                        mask[i,j] = -1e9
        print(mask)

        all_heads = []
        for h in range(n_heads):
            head_score = np.dot(Q[h], np.transpose(K[h])) # [cxt_len , cxt_len]
            head_score /= np.sqrt(self.d_k)
            head_score += mask
            head_score = softmax(head_score)
            head_score = np.matmul(head_score, V[h])
            all_heads.append(head_score)

        scores = np.concatenate(all_heads, axis=-1)
        scores = np.matmul(scores, W_O)
        print("shape of scores: ", np.shape(scores))
        return scores

    def groupedqueryattn(self, x, n_heads, n_groups):

        if n_heads % n_groups != 0:
            raise Exception("Number of heads must be divisible by Number of KV Groups !")

        cxt_len = seq_len(x)
        emb_vec = embed(x)  # input -> embd_vec

        print("cxt_len: ", cxt_len)
        print("embed vec len: ", np.shape(emb_vec))

        W_Q = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)  # embed_dim , n_heads * dk
        W_K = np.random.rand(len(emb_vec[0]), n_groups * self.d_k)
        W_V = np.random.rand(len(emb_vec[0]), n_groups * self.d_k)
        W_O = np.random.rand(n_heads * self.d_k, len(emb_vec[0]))

        Q = np.matmul(emb_vec, W_Q).reshape(cxt_len, n_heads, self.d_k).transpose(1,0,2)
        K = np.matmul(emb_vec, W_K).reshape(cxt_len, n_groups, self.d_k).transpose(1,0,2)
        V = np.matmul(emb_vec, W_V).reshape(cxt_len, n_groups, self.d_k).transpose(1,0,2)

        all_heads = []
        skv_index = 0
        for h in range(n_heads):
            if h > 0 and h % (n_heads // n_groups) == 0:
                skv_index += 1
            head_score = np.dot(Q[h], np.transpose(K[skv_index]))
            head_score /= np.sqrt(self.d_k)
            head_score = softmax(head_score)
            head_score = np.matmul(head_score, V[skv_index])
            all_heads.append(head_score)
        
        scores = np.concatenate(all_heads, axis = -1)
        scores = np.matmul(scores, W_O)
        print("shape of scores: ", np.shape(scores))
        return scores


attn = Attn(d_k=64)
scos = attn.groupedqueryattn("the cat is also called a neko chan", n_heads=8, n_groups=2)
print(scos)



"""

   
    def linearattn(self):
        pass

    def multiqueryattn(self):
        pass

    def multiheadlatentattn(self):
        pass


"""


