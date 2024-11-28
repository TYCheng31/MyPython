import psycopg2

conn = psycopg2.connect(
    dbname="cmsdb",  
    user="cmsuser",            
    password="cmsuser",     
    host="localhost",           
    port="5432"                  
)

cur = conn.cursor()

# 查詢 users 資料
cur.execute("SELECT id, first_name FROM users")
users_data = cur.fetchall()

# 用於記錄已經處理過的 first_name
processed_names = set()

for result in users_data:
    user_id = result[0]
    user_name = result[1]
    
    # 只有第一次遇到該 first_name 時才處理
    if user_name not in processed_names:
        processed_names.add(user_name)  # 標記該 first_name 為已處理
        
        # 查詢 participations 資料
        cur.execute("SELECT id FROM participations WHERE user_id = %s", (user_id,))
        participation_id_result = cur.fetchall()

        if participation_id_result:  # 確保參與記錄存在
            submission_ids = set()  # 用來儲存所有唯一的 submission_id

            for result2 in participation_id_result:
                participation_id = result2[0]  # 提取單個 participation_id
                
                # 查詢 submissions 資料，同時獲取 submission_id 和 task_id
                cur.execute("SELECT id, task_id FROM submissions WHERE participation_id = %s", (participation_id,))
                submission_id_result = cur.fetchall()

                if submission_id_result:  # 如果有對應的 submission_id
                    for submission in submission_id_result:
                        submission_ids.add(submission[0])  # 將每個 submission_id 加入集合，確保唯一性

            # 確保 submission_ids 不為空，並打印每個 submission_id 和對應的 timestamp
            for submission_id in submission_ids:
                # 查詢對應的 timestamp 和 task_id
                cur.execute("SELECT timestamp, task_id FROM submissions WHERE id = %s", (submission_id,))
                submission_result = cur.fetchone()

                cur.execute("SELECT score FROM submission_results WHERE submission_id = %s", (submission_id,))
                score_result = cur.fetchone()
                score = score_result[0]

                if score == 0.0:
                    continue

                if submission_result:  # 確保找到了對應的 timestamp 和 task_id
                    timestamp = submission_result[0]
                    task_id = submission_result[1]
                    # 打印出每個 submission_id 和對應的 timestamp, task_id 和 score
                    print(f"first_name: {user_name} user_id: {user_id} submission_id: {submission_id} timestamp: {timestamp} task_id: {task_id} score: {score}")
                else:
                    print(f"timestamp or task_id not found for submission_id: {submission_id}")
        else:
            print(f"participation_id not found for user_id: {user_id}")

# 關閉游標和資料庫連接
cur.close()
conn.close()
