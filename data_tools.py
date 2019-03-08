import os
import pandas as pd




def import_dataset():
	print("Beginning import of dataset")

	files = []
	for i in os.listdir(os.getcwd() + '/data/'):
	    if i.endswith('.csv'):
	        files.append(open('data/'+ i))

	combined_csv = pd.concat( [ pd.read_csv(f) for f in files ] )
	print("Completed import of dataset")
	return combined_csv

