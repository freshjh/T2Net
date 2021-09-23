import os

src_root = './raw.txt'
tgt_txt = './output.txt'
with open(src_root, 'r') as txt:
    lines = txt.readlines()

Err_count = 0
Info_count = 0

count = 0

for line in lines:

    if '----' in line:
        continue

    # count += 1
    if 'Error' in line:

        Err_count += 1
        text = line.replace('Error:', '')
        content = month + '|' + code + '|' + date + '|' + text + '|'  # need to fix

        with open(tgt_txt, 'a') as tgt:
            tgt.write('Error:')
            tgt.write(content)

    elif 'Info' in line:

        Info_count += 1
        text = line.replace('Info:', '')
        content = month + '|' + code + '|' + date + '|' + text + '|'

        with open(tgt_txt, 'a') as tgt:
            tgt.write('Error:')
            tgt.write(content)

    else:


        code = line.split('|')[0]
        month = line.split('|')[1]
        date = line.split('|')[2][:line.find('+')]


with open(tgt_txt, 'a') as tgt:
    tgt.write('Error count:', Err_count)
    tgt.write('Info count:', Info_count)



