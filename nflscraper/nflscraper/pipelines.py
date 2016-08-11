import csv

class CsvWriterPipeline(object):

	def __init__(self):
		self.csvwriter = csv.writer(open('items.csv','wb'))

	def process_item(self,item,pfr):
		self.csvwriter.writerow([item[key] for key in item.keys()])
		return item