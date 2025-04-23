def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

while True:
    try:
        # 用户输入
        precision = float(input("请输入精确率（Precision）: "))
        recall = float(input("请输入召回率（Recall）: "))

        # 计算F1-score
        f1_score = calculate_f1_score(precision, recall)

        # 输出结果
        print(f"F1-score 是: {f1_score:.4f}")
    except ValueError:
        print("请输入有效的数字。")
    except KeyboardInterrupt:
        print("\n程序已中断。")
        break