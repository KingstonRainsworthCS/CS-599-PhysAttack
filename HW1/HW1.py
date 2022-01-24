#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
from numba import jit, njit
from tqdm import tqdm


# 1. Compute output after the first SubBytes:

# In[2]:


hex(0x000000000000C1A551F1EDC0FFEEB4BE ^ 0x000001020304DECAF0C0FFEE00000000)


# 2. Your coworker needs to implement the AES MixColumns operation in software. He found some code on stackexchange.com. You remember the best practice to review (and test) code before using it in production. Therefore, you offer to review the code. (10 pts)

# In[3]:


# unsigned char MixColumns_Mult_by2(unsigned char Input) {
#     unsigned char Output = Input << 1;
#     if (Input & 0x80)
#         Output ^= 0x1b;
#     return (Output);
# }
#
# unsigned char MixColumns_Mult_by3(unsigned char Input) {
# unsigned char Output = Mult2(Input) ^ Input;
#
#     return (Output);
# }


# 3. Timing attacks:
# 
# Data prepping:

# In[4]:


PATH_TO_KEYS = "./timing_noisy.csv"
keys_data = pd.read_csv(PATH_TO_KEYS, header = None)
sbox = np.array([
        0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
        0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
        0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
        0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
        0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
        0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
        0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
        0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
        0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
        0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
        0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
        0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
        0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
        0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
        0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
        0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])
keys_data_np = keys_data.to_numpy().transpose()
timing_data = keys_data[16].values
sbox_MSB = []
for i in sbox:
    sbox_MSB.append(int(i>=128))
np_sbox_MSB = np.array(sbox_MSB)


# In[9]:


## Naive attempts for one key byte
def naiveCalcBytes():
    key_byte_i = 0
    plaintext = keys_data.head(500000)[key_byte_i].values
    key_candidates = np.arange(0,0x100)
    key_candidates_col=[]
    for i in key_candidates:
        key_candidates_col.append([key_candidates[i]])
    xor_result = pd.DataFrame(np.bitwise_xor(plaintext,key_candidates_col))
    sbox_MSB_search = lambda x: sbox_MSB[x]
    sbox_MSB_result = xor_result.applymap(sbox_MSB_search)

    def calc_time_diff(key):
        time_for_key = pd.DataFrame({'MSB': sbox_MSB_result.iloc[key], 'Time': keys_data[16]})
        result = time_for_key[time_for_key['MSB'] == 1]['Time'].mean() - time_for_key[time_for_key['MSB'] == 0]['Time'].mean()
        return abs(result)

    key_val = 0
    max_diff = 0

    for i in tqdm(key_candidates):
        time_diff = calc_time_diff(i)
        if time_diff > max_diff:
            max_diff = time_diff
            key_val = i

    print({"key_bytes": key_val, "time_diff": max_diff})
start = time.time()
naiveCalcBytes()
end = time.time()
print((end - start) / 60)


# In[26]:


## generalization and optimized
key_candidates = np.arange(0,0x100)
key_candidates_col = []
for i in key_candidates:
    key_candidates_col.append([i])

@njit    
def calc_time_diff(key, xor_result, time_value):
    sbox_MSB_array = np_sbox_MSB[xor_result]
    result = np.mean(time_value[sbox_MSB_array == 1]) - np.mean(time_value[sbox_MSB_array == 0])
    return abs(result)

def calcBytes(num_of_bytes, sample_size=1000000):
    result = []
    for i in range(num_of_bytes):
        key_byte_i = i
        plaintext = keys_data_np[key_byte_i][:sample_size]
        xor_result_op = np.bitwise_xor(plaintext,key_candidates_col)
        
        key_val = 0
        max_diff = 0
        for j in key_candidates:
            time_diff = calc_time_diff(j, xor_result_op[j], timing_data)
            if time_diff > max_diff:
                max_diff = time_diff
                key_val = j
        result.append({"key_bytes": i, "key_val": key_val, "time_diff": max_diff})
    return result


# a. Retrieve key:

# In[40]:


print(calcBytes(16))


# b. Samples size needed:

# In[39]:


## optimized for sample size:
print(calcBytes(1,1000000)) # original
print(calcBytes(1,500000))  # half point
print(calcBytes(1,250000))  # quarter point
print(calcBytes(1,125000))  # 1/8
print(calcBytes(1,62500))   # first false
print(calcBytes(1,100000))  # round to nearest 100k
print(calcBytes(1,110000))  # increment by 10k
print(calcBytes(1,115000))  # increment by 5k
print(calcBytes(1,120000))  # increment by 5k
print(calcBytes(1,121000))  # increment by 1k


# c. Optimization showcases and method

# In[42]:


start = time.time()
res = calcBytes(16,250000)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))


# The bottleneck seems to lies around the SBox lookup in naive attempts. Thus, I replace the regular apply map method to retrieve MSB from Sbox given the xor result by direct array indexing using numpy:

# In[43]:


#sbox_MSB_search = lambda x: sbox_MSB[x]
#sbox_MSB_result = xor_result.applymap(sbox_MSB_search)


# to

# In[44]:


#sbox_MSB_array = np_sbox_MSB[xor_result]


# Similarly, I replace pandas frame / series acess with numpy array and also try to use numpy function for mean.
# After all the optimization, the time took for one key bytes reduce from 2minutes to around 3 second. With near-optimal sample size, the entire thing takes around 11 seconds!
