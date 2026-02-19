我需要你修改@run_chem.py 的脚本，让他能作为baseline生成测试结果。
数据为@data_clean/test_balance_1.csv
测试的指标为Accuracy(预测出的下一个分子的分子式准确率)和Missing Rate(预测出的分子式是否合法，可以遍历一遍test dataset,统计一下所有有效的分子式，便于评估时比对).
可以给chemDFM输入prompt,可以参考@prompt_ref.txt
