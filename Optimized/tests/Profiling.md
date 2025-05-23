# Profiling
**All algorithms were profiled with an input array size of 32768 since that was the limit of the unoptimized kernel.**

## unoptimizedKernel.cu
![image](https://github.com/user-attachments/assets/511e76ee-0648-4834-8bef-0bb86dd8a456)
![image](https://github.com/user-attachments/assets/9186db21-29c9-4249-9b58-e114a52e5d43)
![image](https://github.com/user-attachments/assets/4f589930-22f3-4561-9618-41cbb4b1c9b0)
![image](https://github.com/user-attachments/assets/28c33fb6-718d-43fd-baae-5f256b71e87a)
![image](https://github.com/user-attachments/assets/254ed7f4-e257-4cce-a269-5fe2e130699d)
![image](https://github.com/user-attachments/assets/ea607e05-a257-4ee9-9192-264b7ada2c77)
![image](https://github.com/user-attachments/assets/0e12f26c-177b-4228-a9c4-6809270fec10)
![image](https://github.com/user-attachments/assets/ce9105eb-9bae-4a9b-aaea-03bc7940be19)
![image](https://github.com/user-attachments/assets/42aafdf1-bb5d-4d6b-b534-0f6030f3ad2b)
![image](https://github.com/user-attachments/assets/7ad58495-df26-4599-b952-1c477a804865)
![image](https://github.com/user-attachments/assets/857b0e0a-a709-4058-b1df-6802196cb9bd)

![image](https://github.com/user-attachments/assets/ae014853-eb35-44ba-ac0d-82ab1e8fcaa1)
![image](https://github.com/user-attachments/assets/9ec79efa-0d72-42ef-b180-e596acfc553f)
![image](https://github.com/user-attachments/assets/6f11d6fa-7ea0-4736-8324-67d6abb7482a)
![image](https://github.com/user-attachments/assets/52fe08aa-fb2e-4a02-8099-b8ff06a5fc4b)
![image](https://github.com/user-attachments/assets/95270524-a7c1-41bb-b716-c62ee0620c9b)
![image](https://github.com/user-attachments/assets/a91783d6-a9e6-4af9-87d5-0c61b8d4537c)
![image](https://github.com/user-attachments/assets/15259dac-fa00-457f-b31f-87a3d0ccca39)
![image](https://github.com/user-attachments/assets/b6f60cc8-d306-4e18-98fa-f574e6f980b5)
![image](https://github.com/user-attachments/assets/f291ca81-6d87-423a-b13f-7d9b21ae7e2e)
![image](https://github.com/user-attachments/assets/8af48708-9e58-4ebf-80a5-a07229d30203)

The Kogge_Stone_scan_kernel is taking up the most time.

## optimizedBlelloch.cu
![image](https://github.com/user-attachments/assets/f02c70d5-ce5d-4f3a-8eb9-8bab57b2e773)
![image](https://github.com/user-attachments/assets/94614944-efa8-4d63-8f99-69b220d7c839)
![image](https://github.com/user-attachments/assets/e28a9381-aaff-4f60-93ec-8e2ee22252b8)
![image](https://github.com/user-attachments/assets/976680f7-93e0-4992-995a-a9c282ad7fd9)
![image](https://github.com/user-attachments/assets/912071b9-f4b3-4a7d-ab56-702fb9df36bd)
![image](https://github.com/user-attachments/assets/86774b36-e904-4fda-9576-de3caa17f985)
![image](https://github.com/user-attachments/assets/7b898a9c-2bcc-46db-8c16-f30a0b126cbd)
![image](https://github.com/user-attachments/assets/f69b076f-df61-4bfa-b295-71cddd92556f)
![image](https://github.com/user-attachments/assets/023c7be4-69ca-4d93-a2d3-42aaeeca610c)
![image](https://github.com/user-attachments/assets/c66781c5-1a00-42b1-8ada-cb2178d23735)
![image](https://github.com/user-attachments/assets/56fa7974-2a0c-43b4-a8c9-26b3e6403f5f)

![image](https://github.com/user-attachments/assets/e3138df0-57af-4b23-9a81-c9ba98cf54c8)
![image](https://github.com/user-attachments/assets/41bbf318-65aa-4d87-aa4e-91f7bb134e3d)
![image](https://github.com/user-attachments/assets/69b7a1e8-f599-47f5-b530-67c76aaf9c57)
![image](https://github.com/user-attachments/assets/eaa75fb0-dc47-4e5a-95ec-c2a8480764f4)
![image](https://github.com/user-attachments/assets/ef242980-8c41-4935-a60f-404c8f3ae183)
![image](https://github.com/user-attachments/assets/d54f0a3e-74df-4bf7-b7c0-448fb3ed935b)
![image](https://github.com/user-attachments/assets/0eee5ae5-a3d3-4d56-9621-88a43f77000e)
![image](https://github.com/user-attachments/assets/db81e610-0001-4399-9c5c-f2d464997f37)
![image](https://github.com/user-attachments/assets/88fa5aa0-8ce7-4d8f-b2b0-cb7c2fb5298c)

The CudaMalloc is taking up the most time.

## optimizedWarpLevelScan.cu

![image](https://github.com/user-attachments/assets/3dcc246f-caa1-4ce6-8026-c6117b2cdee0)
![image](https://github.com/user-attachments/assets/70fc81fd-22d9-480e-abf6-87ab2bada809)
![image](https://github.com/user-attachments/assets/0de06f12-b049-482e-a06c-42fb8831b43a)
![image](https://github.com/user-attachments/assets/4df70e35-77f6-4601-b10c-53699b2bf4f5)
![image](https://github.com/user-attachments/assets/3296a51b-7ee7-4d01-a4e5-6a09fcd9f5bb)
![image](https://github.com/user-attachments/assets/81088039-ec54-4931-9d71-e4771045bb31)
![image](https://github.com/user-attachments/assets/10d85342-4132-4cee-846e-32cc26f98796)
![image](https://github.com/user-attachments/assets/93bf258f-26f3-4b6a-b9c4-e8472751c02d)
![image](https://github.com/user-attachments/assets/9f3764d6-be68-4f8b-aacb-984f40782991)
![image](https://github.com/user-attachments/assets/29b1bf0d-ee96-498f-a687-613771010a41)
![image](https://github.com/user-attachments/assets/505fda98-34b6-4dbd-a6c3-a8622c146802)

![image](https://github.com/user-attachments/assets/d47d8ded-7077-4944-9960-abb28a13b581)
![image](https://github.com/user-attachments/assets/ea65f58b-a8b9-47ca-8f78-82fe4c210ece)
![image](https://github.com/user-attachments/assets/9b4edd0a-4ecc-409f-8ee8-13a43e3d430c)
![image](https://github.com/user-attachments/assets/d9ff174e-373c-4ffa-b35a-d6d99c39a5e0)
![image](https://github.com/user-attachments/assets/af428739-6d7c-4653-bc67-d6ee8a398de3)
![image](https://github.com/user-attachments/assets/5b93e8d4-0647-4cbd-9b06-1803ffbc079d)
![image](https://github.com/user-attachments/assets/e109e7f2-f237-47e2-8340-33285a41ff7b)
![image](https://github.com/user-attachments/assets/370559b9-29bc-47e9-868d-5497a1b7ff93)

The Kogge_Stone_scan_kernel is taking the most time.
