import json
import os

# 获取脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 LAT.json 的路径
lat_json_path = os.path.join(script_dir, 'LAT/label/LAT.json')

# 读取 LAT.json
with open(lat_json_path, 'r') as f:
    lat_data = json.load(f)

# 提取所有单品（question 和 answers 里的）
dataset = []  # 格式：[(图片文件名, 类别), ...]

# 定义归类规则
def get_category(item_name):
    if item_name.startswith('Shoes'):
        return 'shoes'
    elif item_name.startswith(('Pants', 'Top', 'Outwear', 'Dress', 'Skirt')):
        return 'clothing'
    elif item_name.startswith('Bags'):
        return 'bags'
    elif item_name.startswith(('Earing', 'Watches', 'Bracelet', 'Hat', 'Sunglasses', 'Neckline')):
        return 'accessories'
    else:
        print(f"未知的单品: {item_name}")
        return 'unknown'

for item in lat_data:
    # 提取 question 里的单品
    for q in item['question']:
        image_name = q.split('_')[1]  # 比如 Pants_P00462138 -> P00462138
        label = get_category(q)
        dataset.append((image_name, label))

    # 提取 answers 里的单品
    for a in item['answers']:
        image_name = a.split('_')[1]  # 比如 Top_P00440101 -> P00440101
        label = get_category(a)
        dataset.append((image_name, label))

# 去重（可能有重复的单品）
dataset = list(set(dataset))

# 统计四大类的分布
class_counts = {'shoes': 0, 'clothing': 0, 'accessories': 0, 'bags': 0, 'unknown': 0}
for _, label in dataset:
    class_counts[label] = class_counts.get(label, 0) + 1

print("类别分布：", class_counts)

# 检查图片文件是否存在
image_dir = os.path.join(script_dir, 'LAT/image/')
missing_images = []
for image_name, label in dataset:
    image_path = os.path.join(image_dir, image_name + '.jpg')  # 比如 P00462138.jpg
    if not os.path.exists(image_path):
        missing_images.append(image_name)

if missing_images:
    print("以下图片文件缺失：", missing_images)
else:
    print("所有图片文件都存在！")

# 保存新的数据集（图片路径和标签）
output_path = os.path.join(script_dir, 'fashion_dataset.json')
with open(output_path, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"数据集已保存到 {output_path}")