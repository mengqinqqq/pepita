import io
from PIL import Image
import sys
import xlsxwriter

import analyze6
import imageops

def make(filenames):
	workbook = xlsxwriter.Workbook('juxtapose.xlsx', {'nan_inf_to_errors': True})
	worksheet = workbook.add_worksheet()

	worksheet.set_column('C:F', 68)

	images = analyze6.quantify(filenames)

	for i in range(len(filenames)):
		worksheet.set_row(i, 275)
		image = images[i]

		worksheet.write(i, 0, image.plate)
		worksheet.write(i, 1, image.well)
		worksheet.write(i, 2, 'XY' + str(image.xy))

		# non-rescaled it's often too dark, rescaled it'll be inaccurate...
		# downscaled_bf_img = imageops.resize(image.bf_img, 0.25)
		# downscaled_fl_img = imageops.resize(image.fl_img, 0.25)
		downscaled_bf_img = imageops.rescale_brightness(imageops.resize(image.bf_img, 0.25))
		downscaled_fl_img = imageops.rescale_brightness(imageops.resize(image.fl_img, 0.25))
		downscaled_mask = imageops.resize(image.mask, 0.25)
		downscaled_bf_img_masked = imageops.apply_mask(downscaled_bf_img, downscaled_mask)
		downscaled_fl_img_masked = imageops.apply_mask(downscaled_fl_img, downscaled_mask)

		worksheet.insert_image(i, 3, 'bf_img', {'image_data': img_to_buffer(downscaled_bf_img)})
		worksheet.insert_image(i, 4, 'fl_img', {'image_data': img_to_buffer(downscaled_fl_img)})
		worksheet.insert_image(
			i, 5, 'bf_img_masked', {'image_data': img_to_buffer(downscaled_bf_img_masked)}
		)
		worksheet.insert_image(
			i, 6, 'fl_img_masked', {'image_data': img_to_buffer(downscaled_fl_img_masked)}
		)

		worksheet.write(i, 7, image.normalized_value)

	workbook.close()

def img_to_buffer(img):
	bytes_io = io.BytesIO()
	Image.fromarray(img).save(bytes_io, format='png')
	return bytes_io

if __name__ == '__main__':
	if len(sys.argv) > 1:
		make(sys.argv[1:])
	else:
		raise TypeError('Invoke with an argument, i.e. the name of a file or files to analyze and put into spreadsheet.')
