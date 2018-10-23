
input_name = 'official_val.txt'
output_name = 'official_val_id.txt'

input_file = open(input_name, 'r')
output_file = open(output_name, 'w')

id_list = []
for line in input_file:
	line = line.strip();
	id_list.append(int(line.split('_')[-1].split('.')[0]))

for i in id_list:
	output_file.write(str(i) + '\n')




