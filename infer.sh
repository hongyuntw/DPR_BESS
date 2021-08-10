python3 generate_dense_embeddings.py \
	--model_file ./results/movieQA_cosine_QA_cn/dpr_biencoder.1.3018 \
	--ctx_file ../DPR_facebook/data/retriever/movie-cn-dpr.tsv \
	--out_file ./ctx_embedding_output/movieQA_cosine_QA_cn  \
	--do_lower_case \
    --batch_size 1000 \

python3 dense_retriever.py \
	--model_file ./results/movieQA_cosine_QA_cn/dpr_biencoder.1.3018  \
	--ctx_file ../DPR_facebook/data/retriever/movie-cn-dpr.tsv \
	--qa_file ../DPR_facebook/data/retriever/movie-cn-test.tsv \
	--encoded_ctx_file ./ctx_embedding_output/movieQA_cosine_QA_cn_0.pkl \
	--out_file ./ctx_embedding_output/movieQA_cosine_QA_cn \
	--do_lower_case \
	--n-docs 100 \
    --max_mask_length 32
