python3 dense_retriever.py \
	--model_file ./results/movieQA_cosine_QA/dpr_biencoder.1.3139  \
	--ctx_file ../DPR_facebook/data/retriever/movie-dpr.tsv \
	--qa_file ../DPR_facebook/data/retriever/movie-test.tsv \
	--encoded_ctx_file ./ctx_embedding_output/movieQA_cosine_QA_0.pkl \
	--out_file ./ctx_embedding_output/movieQA_cosine_QA \
	--do_lower_case \
	--n-docs 100 \
    --max_mask_length 32
