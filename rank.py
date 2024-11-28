import csv
from collections import defaultdict

# 用來儲存每個使用者的總分和最後提交時間
user_scores = defaultdict(lambda: {'total_score': 0.0, 'first_name': '', 'last_timestamp': ''})

# 讀取 CSV 檔案
with open('output.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)

    # 逐行讀取資料，並累加每位 user 的分數
    for row in reader:
        first_name = row['first_name']
        highest_score = float(row['highest_score'])
        timestamp = row['timestamp']

        # 累加該 user 的分數並保留最後一次提交的時間
        user_scores[first_name]['total_score'] += highest_score
        user_scores[first_name]['first_name'] = first_name
        
        # 更新最後一次提交的時間（只保留最新的時間）
        if user_scores[first_name]['last_timestamp'] < timestamp:
            user_scores[first_name]['last_timestamp'] = timestamp

# 輸出每位使用者的總分和最後提交時間
for data in user_scores.values():
    print(f"Full Name: {data['first_name']} | Total Score: {data['total_score']} | Last Submission Time: {data['last_timestamp']}")
