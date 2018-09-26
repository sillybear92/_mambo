import os
import glob
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
from os.path import isfile, join

def conv_xml(xml_path,img_path,save_path):
	xml_list=[]
	for xml_file in glob.glob(xml_path + '\\*.xml'):
		r_tree = ET.parse(xml_file)
		r_root = r_tree.getroot()
		boxes=[]
		root=ET.Element('annotation')
		tree=ET.Element(root)

		folder=ET.Element('folder')
		folder.text='imgs\\'
		root.append(folder)

		try:
			img_filename=xml_file.split('\\')[6].split('.')[0]+'.'+(r_root.find('filename').text).split('.')[2]
		except:
			img_filename=xml_file.split('\\')[6].split('.')[0]+'.'+(r_root.find('filename').text).split('.')[1]
		#img_filename=r_root.find('filename').text
		xml_filename=img_filename.split('.')[0]+'.xml'
		filename_node = ET.Element('filename')
		filename_node.text = img_filename
		root.append(filename_node)

		filepath_node = ET.Element('path')
		filepath_node.text=filename_node.text
		root.append(filepath_node)
		print(img_filename)
		i_path = join(img_path, img_filename)
		img= cv2.imread(i_path)
		img_size=img.shape
		size_node=ET.Element('size')
		width_node=ET.Element('width')
		width_node.text=str(img_size[1])
		size_node.append(width_node)

		height_node = ET.Element('height')
		height_node.text = str(img_size[0])
		size_node.append(height_node)
		
		depth_node=ET.Element('depth')
		depth_node.text=str(img_size[2])
		size_node.append(depth_node)
		root.append(size_node)

		segmented_node = ET.Element('segmented')
		segmented_node.text = '0'
		root.append(segmented_node)

		for member in r_root.findall('object'):
				object_node=ET.Element('object')
				# classes Name
				# ob_name_node.text = 'bicycle'
				ob_name_node=ET.Element('name')
				ob_name_node.text=member[0].text
				object_node.append(ob_name_node)
				pose_node = ET.Element('pose')
				pose_node.text = 'Unspecified'
				object_node.append(pose_node)
				truncated_node=ET.Element('truncated')
				truncated_node.text='0'
				object_node.append(truncated_node)
				difficult_node=ET.Element('difficult')
				difficult_node.text='0'
				object_node.append(difficult_node)
				box_node=ET.Element('bndbox')
				xmin_node=ET.Element('xmin')
				xmin_node.text=member[4][0].text
				box_node.append(xmin_node)
				ymin_node=ET.Element('ymin')
				ymin_node.text=member[4][1].text
				box_node.append(ymin_node)
				xmax_node=ET.Element('xmax')
				xmax_node.text=member[4][2].text
				box_node.append(xmax_node)
				ymax_node=ET.Element('ymax')
				ymax_node.text=member[4][3].text
				box_node.append(ymax_node)
				object_node.append(box_node)
				root.append(object_node)
				cv2.rectangle(img,(int(member[4][0].text),int(member[4][1].text)),(int(member[4][2].text),int(member[4][3].text)),(0,255,0),2)
				
		rough_xml= ET.tostring(root,'utf-8')
		rough_xml=minidom.parseString(rough_xml)
		pretty_xml = rough_xml.toprettyxml()
		
		xml_path= join(save_path, xml_filename)
		with open(xml_path,'w') as xml_file:
			xml_file.write(pretty_xml)
		cv2.imshow(img_filename,img)
		cv2.waitKey(1)
		cv2.destroyAllWindows()
#your input xml_path 
xml_path = os.path.join('F:\\AnacondaProjects\\detect_data\\FREEVI~1\\bollard\\320')
#your img_path
img_path = os.path.join('F:\\AnacondaProjects\\detect_data\\image')
#your output xml_path
save_path=os.path.join('F:\\AnacondaProjects\\detect_data\\annotations')
conv_xml(xml_path, img_path,save_path)
