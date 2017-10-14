#reduce size of user logs step 
count = 0
for chunk in pd.read_csv('resources/user_logs_sum_all.csv', names=['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'], chunksize=10000000):
    df_temp = chunk.groupby('msno')['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'].sum()
    df_temp.to_csv('resources/user_logs_sum_all2.csv', mode='a', header=False)
    count += 1
    print(count)
    
#step 2 read entire file into memory and sum columns 
df = pd.read_csv('resources/user_logs_sum_all2.csv', names=['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'])
df = df.groupby('msno')['num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'].sum()
df.to_csv('resources/user_logs_sum_all3.csv', mode='a', header=False)

#step 3 add column names
df = pd.read_csv('resources/user_logs_sum_all3.csv', names=['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'])
df.to_csv('resources/user_logs_final.csv', columns=['msno', 'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq', 'total_secs'])
