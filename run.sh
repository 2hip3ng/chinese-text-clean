# --jieba_vocab_file 自定义词表位置
# --stopwords_file  停用词表位置


python data_clean.py \
		   --task_name demo \
		   --input_file ./test.txt \
		   --output_file ./test.txt.clean \
		   --rm_url true \
		   --rm_unknown_char true \
		   --jieba_cut true \
		   --stopwords_file ./哈工大停用词表.txt
		   
