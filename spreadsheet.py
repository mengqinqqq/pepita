import xlsxwriter
import analyze6

def make(filenames):
	workbook = xlsxwriter.Workbook('juxtapose.xlsx')
	worksheet = workbook.add_worksheet()

	# pd dataframe of (plate, column, row, normalized_value)
	normalized_results = analyze6.quantify(filenames)

	# worksheet.set_column('C:C', 30) # widen
	# worksheet.set_column('D:D', 30) # widen

	for i in range(len(filenames)):
		row = normalized_results.iloc[i]

		worksheet.write(i + 1, 0, row['plate'])
		worksheet.write(i + 1, 1, row['well'])
		worksheet.insert_image(i + 1, 2, scaled_down_bf_img)
		worksheet.insert_image(i + 1, 3, scaled_down_fl_img)
		worksheet.insert_image(i + 1, 4, masked_scaled_down_bf_img)
		worksheet.insert_image(i + 1, 5, masked_scaled_down_fl_img)
		worksheet.write(i + 1, 6, row['value'])
		# H:H - (manual) bf acceptable (-1, 0, 1)
		# I:I - (manual) fl acceptable (-1, 0, 1)

	workbook.close()

if __name__ == '__main__':
	if len(sys.argv) > 1:
		make(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze and put into spreadsheet.')
