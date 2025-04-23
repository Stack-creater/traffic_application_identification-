# 打开文本文件并读取数据
with open('hs_resources/rulesets.txt', 'r', encoding='utf-8') as file:
    data = file.readlines()
print(data)
# 创建一个空字典来存储结果
word_dict = {}

# 遍历每行数据并解析
for line in data:
    line = line.strip()
    if line:
        parts = line.split(' ')
        if len(parts) == 3:
            weight, word, category = parts
            weight = float(weight)
            category = category.strip()

            # 如果权值大于0，则将其添加到相应的类别列表
            if weight > 0:
                if word in word_dict:
                    word_dict[word].append(f'{category}:{weight}')
                else:
                    word_dict[word] = [f'{category}:{weight}']
print(len(word_dict))
# 打印结果
for word, categories in word_dict.items():
    print(f"{word}: {categories}")
