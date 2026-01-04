"""
    ╱|、
    (˚ˎ 。7  
    |、˜〵          
    じしˍ,)ノ

    kittention is made out of love, not for production usage 
"""


import numpy as np
import scipy as sp
from sentence_transformers import SentenceTransformer
from utils import seq_len, softmax, softplus, ReLU, logsumexp, gaussian_ppf


np.random.seed(42)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed(text):
    emb_vec = np.array([model.encode(token) for token in text.split(" ")])
    return emb_vec                                    # (num_words , embed dim)


class Kittention:
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
                        print(f"given i = {i}, j = {j} ; and j < = i {cond1} with j > = i - local_window {cond2} or j % stride is {cond3}")
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
    
    def sparsesinkhornattn(self, x, n_heads, block_size, n_sinkhorn_iters=5, temperature=1.0, use_gumbel=True, sortcut_k=None):
        """https://arxiv.org/pdf/2002.11296"""

        cxt_len = seq_len(x)
        emb_vec = embed(x)  # input -> embd_vec
        n_blocks = cxt_len // block_size

        W_Q = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)  # embed_dim , n_heads * dk
        W_K = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_V = np.random.rand(len(emb_vec[0]), n_heads * self.d_k)
        W_O = np.random.rand(n_heads * self.d_k, len(emb_vec[0]))

        # (cxt_len, n_heads * dk) -> (cxt_len, n_heads, dk) -> (n_blocks, block_size, n_heads, dk) -> (n_heads, n_blocks, block_size, dk)
        Q = np.matmul(emb_vec, W_Q).reshape(cxt_len, n_heads, self.d_k).reshape(n_blocks, block_size, n_heads, self.d_k).transpose(2,0,1,3)
        K = np.matmul(emb_vec, W_K).reshape(cxt_len, n_heads, self.d_k).reshape(n_blocks, block_size, n_heads, self.d_k).transpose(2,0,1,3)
        V = np.matmul(emb_vec, W_V).reshape(cxt_len, n_heads, self.d_k).reshape(n_blocks, block_size, n_heads, self.d_k).transpose(2,0,1,3)


        # step 2: create SORTNET
        all_heads = []
        for h in range(n_heads):
            block_summary_k = np.sum(K[h], axis=1)

            hidden_dim = 64
            W1 = np.random.rand(self.d_k, hidden_dim)
            b1 = np.random.rand(hidden_dim)
            W2 = np.random.rand(hidden_dim, n_blocks)
            b2 = np.random.rand(n_blocks)

            h1 = ReLU(np.matmul(block_summary_k, W1) + b1) 
            log_alpha = np.matmul(h1, W2) + b2

            if use_gumbel: 
                gumbel = -np.log(-np.log(np.random.uniform(size=log_alpha.shape) + 1e-10) + 1e-10)
                log_alpha = (log_alpha + gumbel) / temperature
            
            # step 3: sinkhorn normalization (matrix becomes doubly stocastic)
            log_P = log_alpha

            for _ in range(n_sinkhorn_iters):
                log_P = log_P - logsumexp(log_P, axis = 1, keepdims=True)
                log_P = log_P - logsumexp(log_P, axis = 0, keepdims=True)
            P = np.exp(log_P)

            print("all rows should sum to 1: " + str(np.sum(P, axis=0))) 
            print("all cols should sum to 1: " + str(np.sum(P, axis=1))) 


            # step 4: permutation matrix
            # first flatten K and V (n blocks, block_size * dk) then matmul then back to (n_blocks, block_size, dk)
            K_sorted = np.matmul(P,K[h].reshape(n_blocks, -1)).reshape(n_blocks, block_size, self.d_k)
            V_sorted = np.matmul(P, V[h].reshape(n_blocks, -1)).reshape(n_blocks, block_size, self.d_k)

            output_blocks = []
            for i in range(n_blocks):
                scores = np.matmul(Q[h][i], K_sorted[i].T) 
                scores /= np.sqrt(self.d_k)
                scores = softmax(scores)
                out = np.matmul(scores, V_sorted[i])  # (block_size, dk)
                output_blocks.append(out)
            head_score = np.concatenate(output_blocks, axis=0)
            all_heads.append(head_score)
        
        scores = np.concatenate(all_heads, axis = 1)
        scores = np.matmul(scores, W_O)
        print("shape of scores: ", np.shape(scores))
        return scores
    
    def sparkattention(self, x, n_heads, k_attn, r=None, use_rope=False):
        """https://arxiv.org/pdf/2506.06644"""

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

        if r is None: r = self.d_k // 2

        # step1: key splitting
        all_heads = []
        for h in range(n_heads):

            word_scores = []
            K1 = np.transpose(K[h][:,:r])  # (num_words , dk) -> (num_words, r) -> (r, num_words)
            K2 = np.transpose(K[h][:,r:]) # (d_k - r, num_words)
            print("shape of K1 : ", np.shape(K1)) 
            print("shape of K2 : ", np.shape(K2)) 
            V_h = np.transpose(V[h])

            for i in range(cxt_len):
                q1 = Q[h, i, :r] # (r) for i
                q2 = Q[h, i, r:] # (d_k - r) for i
              
                # step2: predictor scores
                casual_K1 = K1[:, :i+1]
                score_pred = np.matmul(q1, casual_K1) # (num_words,)
                
                # step3: topk selection
                mu = np.mean(score_pred)
                sigma = np.std(score_pred)
                p = 1.0 - (k_attn / len(score_pred))
                z = gaussian_ppf(p)
                θ = mu + sigma * z

                keep = score_pred > θ
                idx = np.where(keep)[0]
                if len(idx) == 0: idx = np.array([len(score_pred)-1])
    
                # step4: sparse softmax for predictor scores
                score_pred_select = score_pred[idx]
                a_idx = softmax(score_pred_select) # predictor attn weights

                # step5: complute value scores for selected tokens
                casual_K2 = K2[:, :i+1]
                score_value = np.array([np.matmul(q2, casual_K2[:,j]) for j in idx])
                b_idx = softplus(score_value)
                
                w_idx = np.multiply(a_idx, b_idx)

                casual_V = V_h[:, :i+1]
                output = np.sum(w_idx[:, None] * np.transpose(casual_V[:, idx]), axis=0)
                word_scores.append(output)
        
            head_output = np.stack(word_scores, axis=0)
            all_heads.append(head_output)
        
        scores = np.concatenate(all_heads, axis=1)  # (num_words, n_heads*d_k)
        scores = np.matmul(scores, W_O)              # (num_words, embed_dim)
        print("shape of scores: ", np.shape(scores))
        return scores



attn = Kittention(d_k=64)
scos = attn.sparkattention("the cat is also called a neko chan", n_heads=1, k_attn=3, r=None, use_rope=False)
print(scos)



