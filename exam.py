import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as mrdb


dfbream_length = pd.read_csv('.\\fish\\bream_length.csv', header=None)
dfbream_weight = pd.read_csv('.\\fish\\bream_weight.csv', header=None)
dfsmelt_length = pd.read_csv('.\\fish\\smelt_length.csv', header=None)
dfsmelt_weight = pd.read_csv('.\\fish\\smelt_weight.csv', header=None)

print("="*50)
print(dfbream_length.shape)
print("="*50)
print("="*50)

# 도미 길이
npbream_length = dfbream_length.to_numpy()
bream_length = npbream_length.flatten()
print(bream_length)

# 도미 무게
npbream_weight = dfbream_weight.to_numpy()
bream_weight = npbream_weight.flatten()
print(bream_weight)

# 빙어 길이
npsmelt_length = dfsmelt_length.to_numpy()
smelt_length = npsmelt_length.flatten()
print(smelt_length)

# 빙어 무게
npsmelt_weight = dfsmelt_weight.to_numpy()
smelt_weight = npsmelt_weight.flatten()
print(smelt_weight)

# 도미 [ [길이,무게], [길이,무게] ... [길이,무게] ]
bream_data = np.column_stack((bream_length, bream_weight))
print(bream_data)
print("="*50)
# 빙어 [ [길이,무게], [길이,무게] ... [길이,무게] ]
smelt_data = np.column_stack((smelt_length, smelt_weight))
print(smelt_data)
print("="*50)

print(bream_data.shape)
print(smelt_data.shape)
print("="*50)

plt.scatter(bream_data[:, 0], bream_data[:, 1])
plt.scatter(smelt_data[:, 0], smelt_data[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
print("="*50)

fish_length = np.concatenate((bream_length, smelt_length))
print(fish_length)
print("="*50)

fish_weight = np.concatenate((bream_weight, smelt_weight))
print(fish_weight)
print("="*50)

print(fish_length.shape)
print(fish_weight.shape)
print("="*50)


fish_data = np.column_stack((fish_length, fish_weight))
print(fish_data)
print("="*50)

# (47,2) - 0 ~ 46
print(fish_data.shape)
print("="*50)

# 도미 1~35 35개 , 빙어 1~14 14개
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
print(fish_target)
print("="*50)

fish_target = fish_target.reshape((-1, 1))
print(fish_data)
print(fish_target)
print("="*50)
print("="*50)
fishes = np.hstack((fish_target, fish_data))
# 랜덤 섞기
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)
print("="*50)

# 훈련 데이터 (모델)
train_input = fish_data[index[:35]]
# 타겟 데이터 (모델)
train_target = fish_target[index[:35]]


# 훈련 데이터 (검증)
test_input = fish_data[index[35:]]
# 타겟 데이터 (검증)
test_target = fish_target[index[35:]]

print("="*50)
print("="*50)
print("="*50)
print("="*50)
print("="*50)
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

engine = mrdb.create_engine(
    "mariadb+mariadbconnector://python:python1234@127.0.0.1:3306/pythondb")

# 9. 훈련 데이터 DB 저장 - 테이블명 : train
train_data = pd.DataFrame(fishes[index[:35]], columns=[
                          "train_target", "train_lentgh", "train_weight"])
train_data.to_sql("train", engine, index=False, if_exists="replace")

# 10. 테스트 데이터 DB 저장 - 테이블명 : test
test_data = pd.DataFrame(fishes[index[35:]], columns=[
                         "test_target", "test_lentgh", "test_weight"])
test_data.to_sql("test", engine, index=False, if_exists="replace")
