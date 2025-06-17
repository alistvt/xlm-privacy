File: cs.txt
  Number of lines processed: 1174893
  Average tokens per line (first 100:100100 lines): 30.43
  Token-length threshold (p=10% exceed): 70
  Token-length threshold (p=20% exceed): 50
  Token-length threshold (p=30% exceed): 37
  Token-length threshold (p=40% exceed): 28
  Token-length threshold (p=50% exceed): 21
  Token-length threshold (p=60% exceed): 15
  Token-length threshold (p=70% exceed): 10
  Token-length threshold (p=80% exceed): 7
  Token-length threshold (p=90% exceed): 3
  Number bigger than 120 token: 19672

File: en.txt
  Number of lines processed: 1201437
  Average tokens per line (first 100:100100 lines): 16.83
  Token-length threshold (p=10% exceed): 36
  Token-length threshold (p=20% exceed): 26
  Token-length threshold (p=30% exceed): 20
  Token-length threshold (p=40% exceed): 16
  Token-length threshold (p=50% exceed): 12
  Token-length threshold (p=60% exceed): 9
  Token-length threshold (p=70% exceed): 7
  Token-length threshold (p=80% exceed): 4
  Token-length threshold (p=90% exceed): 3
  Number bigger than 64 token: 20019

File: nl.txt
  Number of lines processed: 1217192
  Average tokens per line (first 100:100100 lines): 25.69
  Token-length threshold (p=10% exceed): 57
  Token-length threshold (p=20% exceed): 41
  Token-length threshold (p=30% exceed): 31
  Token-length threshold (p=40% exceed): 25
  Token-length threshold (p=50% exceed): 19
  Token-length threshold (p=60% exceed): 13
  Token-length threshold (p=70% exceed): 9
  Token-length threshold (p=80% exceed): 6
  Token-length threshold (p=90% exceed): 3
  Number bigger than 100 token: 18955

File: pl.txt
  Number of lines processed: 1212815
  Average tokens per line (first 100:100100 lines): 32.15
  Token-length threshold (p=10% exceed): 74
  Token-length threshold (p=20% exceed): 52
  Token-length threshold (p=30% exceed): 39
  Token-length threshold (p=40% exceed): 29
  Token-length threshold (p=50% exceed): 22
  Token-length threshold (p=20% exceed): 18
  Token-length threshold (p=30% exceed): 10
  Token-length threshold (p=40% exceed): 10
  Token-length threshold (p=50% exceed): 10
  Token-length threshold (p=60% exceed): 5
  Token-length threshold (p=70% exceed): 5
  Token-length threshold (p=80% exceed): 1
  Token-length threshold (p=90% exceed): 1
  Number bigger than 120 token: 28112

File: sl.txt
  Number of lines processed: 1181005
  Average tokens per line (first 100:100100 lines): 28.10
  Token-length threshold (p=10% exceed): 64
  Token-length threshold (p=20% exceed): 45
  Token-length threshold (p=30% exceed): 34
  Token-length threshold (p=40% exceed): 26
  Token-length threshold (p=50% exceed): 19
  Token-length threshold (p=60% exceed): 14
  Token-length threshold (p=70% exceed): 10
  Token-length threshold (p=80% exceed): 6
  Token-length threshold (p=90% exceed): 3
  Number bigger than 108 token: 22382


File: en-sl
  Total tokens: 18307106
  Total tokens: 30900571
  RATIO: 1.687900370490016

File: en-nl
  Total tokens: 18983728
  Total tokens: 29122669
  RATIO: 1.534085876072392

File: en-pl
  Total tokens: 18352587
  Total tokens: 35708155
  RATIO: 1.945674198411374

File: en-cs
  Total tokens: 18440945
  Total tokens: 33570225
  RATIO: 1.820417825659151


SO we use the following context length to preprocess files:
  en: 64
  sl: 108
  nl: 98
  pl: 124
  cs: 116