import csv
from collections import defaultdict

# 用來儲存每個使用者的總分
user_scores = defaultdict(float)

# 讀取 CSV 檔案
with open('output.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # 逐行讀取資料，並累加每位 user 的分數
    for row in reader:
        user_id = row['user_id']
        first_name = row['first_name']
        highest_score = float(row['highest_score'])

        # 累加該 user_id 的最高分數
        user_scores[user_id] += highest_score

# 輸出每位使用者的總分
for first_name, total_score in user_scores.items():
    print(f"User ID: {first_name} | Total Score: {total_score}")
