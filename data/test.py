import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'


from datasets import load_dataset


# 加载 Restaurants 数据集
restaurants = load_dataset("semeval2014_absa", "restaurants")
# 加载 Laptops 数据集
laptops = load_dataset("semeval2014_absa", "laptops")

# 餐厅领域 (Restaurants)
dataset_rest = load_dataset("semeval2014_absa", "restaurants")
# 笔记本领域 (Laptops)
dataset_lap = load_dataset("semeval2014_absa", "laptops")

# 保存餐厅数据
dataset_rest.save_to_disk(os.path.join(".\raw", "semeval2014_restaurants"))
# 保存笔记本数据
dataset_lap.save_to_disk(os.path.join(".\raw", "semeval2014_laptops"))