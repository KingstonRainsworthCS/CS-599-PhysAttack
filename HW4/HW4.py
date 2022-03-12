#!/usr/bin/env python
# coding: utf-8

# In[70]:


import numpy as np


# ## 1. RSA clock speed requirement
# 
# An important processing platform for digital signatures are smartcards. Older smartcards
# are typically based on an 8-bit processor. Let us assume that such a processor requires one
# clock cycle for an 8x8 bit multiply-and-add. Please assume that a simple square-and-multiply
# algorithm is used and a b-bit multiplication and a b-bit squaring requires the same number
# of clock cycles. (30 pts)

#     a. What is the minimum processor clock speed to process an RSA signature in less than 0.75s if modulus N is 512 bits? 1024 bits?

# Assuming that we only need to perform the verification of RSA signature (no hash), we expected to do one modulus exponential c^d and one comparision under mod N. If d is uniformly chosen (equal chance of 0s and 1s). Also assume no CRT implementation:

# In[71]:


## b = 512
# Number of "group" that an 8 bits processor will have to process:
group512 = (512**2)/64
print("Groups #: ", group512)
# Out of those "group", we expected on average half would contain a multiply operation, meaning 4 in each group of 8.
cycles512 = (0.5*1+0.5*2)*512*group512
print("Number of cycles needed: ", cycles512)
print("Processor needed: ", cycles512 / 0.75)


# In[72]:


## b = 1024
# Number of "group" that an 8 bits processor will have to process:
group1024 = (1024**2)/64
print("Groups #: ", group1024)
# Out of those "group", we expected on average half would contain a multiply operation, meaning 4 in each group of 8.
cycles1024 = (0.5*1+0.5*2)*1024*group1024
print("Number of cycles needed: ", cycles1024)
print("Processor needed: ", cycles1024 / 0.75)


#     b. Is this a reasonable goal if the processor is not allowed to clock faster than 8 MHz? (5 pts)

# Based on our number it seems like this is more than enough for 512 bit case but is rather lacking for 1024. This can be further optimize by making sure that our calculation doesn't overflow the register using CRT

# ## 2. Fault Injection Attack

#     a. Bellcore's method

# In[73]:


n = int("9B1F16A7696AC90FA7AE615A1F71BD1AC0C31B37A9F14376BEC7FB701412F0E3B79CAB88F906B350B521578766C78CACD2E80632D0935F50CDDC415DC1B046EB3B3556624EB412D056F873E5A056C2B85B364D032BBAA9276757B058879B02CB2098D63C61551D4753B1AE1890D8FF79BA10F82307492A775A5715AD605B5601", 16)
s = int("1448FA660D3DEE693A9AE10E3DBE176DD0AE9637F2896003367FAB3F71C1BE2C8A143DD9E4167C9A07801E666AC268F376EE6B9A27752322E1BBD16F8DDA2B90058A07B1AB564537C800953BD23771D8CFD08B96D34BA6013B10383B19D0F263E0EEBC7D09FDFEA003D73DDA885D25A3C2870CF8E5FFE7201AE75874F383097B", 16)
sprime = int("9ABD38BDD461B4F8F6A1824B1B43D41C18319071CA6028865576C32A258532BB08A449F29372F4BDB016B5A1F57AAA96BE66B17ECF0CD2BF89C7CAC77E5B1A43460688C3EDBF6DA6EBCEEA9B5797A4FC28EC93EF18DA6EE54A523861ABEDC82C4A148EC5C88DE1C51B6C813C8C13173E8526D0035E2A375CF7222A18C2860B1A", 16)


# In[74]:


## get factor:
p = np.gcd(s-sprime,n)
## check if p and q =1/p is factor:
print(np.mod(n,p))
q = n//p
print(np.mod(n,q))


# In[75]:


print("Prime factors: ")
print("q: ", q)
print("p: ", p)


#     b. Lentra's method

# In[76]:


nL = int("C2D2BE8E722AE5BBD23DFAD362A08B4D32A45115542E23E49B3546583338CD8B8BA42EF289B2E447E9BF6EAF7F24D02565D224ABDDD6D2F44A6F2816A4323196942DF20DED8F10024524E1B2F02F4AD0C1CBF7C778270BCD708EBFA049384EDEEF24C084DA3CA2EE146CA579CC42AEE7F6D4B0F59E5843A519329BEB5F976607", 16)
e = int("010001", 16)
m = int("1E8ADB08E98A58012C55A8C419747BD8D8DB40FAC240DA92BF4874F79E9AD73B20A934070CAA60C767254168ABEB37955618458F6BF94B2D7BA8921DE7E84FA67AF7E0D6FE9EDD554ABF4418F7AEE8D829E6EC1245CFCBAF589667963B531B89AF63879C9A653176A03BA689BC5DD45DA663910A19FA496A6AEFB3F9ADFFF696", 16)
sprimeL = int("2572EE15579D2E18724E98A137BC82CC46654E04E0AF227C36D7B0C29EF49D1B7757A367712EBC6C8DAD7E526678860CCD44AFFBE0C3791F4E0BA3E1863303E807CC4BD8A89542B22158D67D99DC93050ACA584D2D06950B6DC6157E47CFED4DC6D877E47A0C7F1A09FEA4115EBF67EFDAF4A8409689054366E58786E74D2ABD", 16)


# In[77]:


## get factor:
qL = np.gcd(nL, m-(pow(sprimeL, e, nL)))
## check if q andp = nL // q is factor:
print(np.mod(nL, qL))
pL = nL // qL
print(np.mod(nL, pL))


# In[78]:


print("Prime factors: ")
print("q: ", qL)
print("p: ", pL)


# In[ ]:




