python3 generate_dense_embeddings.py \
	--model_file results/msmarco_query_augment/dpr_biencoder.1.14963 \
	--ctx_file ../DPR_facebook/data/retriever/msmarco-collections.tsv \
	--out_file ./ctx_embedding_output/msmarco_query_augment  \
	--do_lower_case \
    --batch_size 2000 \
