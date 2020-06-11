import xml.sax
import xml.dom.minidom
from xml.dom.minidom import getDOMImplementation
import cv2
import os
'''
对D2-City的每个视频和对应的标注文件：
第一步：将标注文件中关于person类的标注提取出来（主要是提取框坐标和第几帧frame信息）
第二步：保存视频中的这几帧，每帧一张图片（或隔几帧保存一次，保存到JPEGImages文件夹中，jpg格式）
第三步：将person类的标注转换成“labelImg”生成的那种xml格式（保存到Annotations文件夹中）
'''
xml_src_path = r'C:/AIoT/D2-City/valid_annotation/0008'  # D2-City xml文件夹
xml_save_path = r'./Annotations'  # 保存xml的文件夹
fileName = ""
videos_src_path = r'C:/AIoT/D2-City/valid_video/0008'  # 视频文件夹
frame_save_path = r'./JPEGImages'  # 保存图片的文件夹


class DataHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.isPerson = False
        self.ybr = ""
        self.xbr = ""
        self.ytl = ""
        self.xtl = ""
        self.frame = ""
        self.dict = {}

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        if tag == "track":
            if attributes["label"] == "person":
                self.isPerson = True
        if tag == "box" and self.isPerson:
            self.ybr = attributes["ybr"]
            self.xbr = attributes["xbr"]
            self.ytl = attributes["ytl"]
            self.xtl = attributes["xtl"]
            self.frame = attributes["frame"]

    # 元素结束事件处理
    def endElement(self, tag):
        if tag == "annotations" and len(self.dict) != 0:
            dict_keys = self.dict.keys()
            cap = cv2.VideoCapture(videos_src_path + "/" + fileName + ".mp4")  # 用OpenCV一帧一帧读取出来
            frame_count = 1
            while True:
                success, frame = cap.read()
                if success:
                    if frame_count in dict_keys:
                        cv2.imwrite(frame_save_path + "/" + fileName + "_%d.jpg" % frame_count, frame)
                        print("Save frame: " + fileName + "_%d.jpg" % frame_count)
                        self.createXML(frame_count)  # 基于[(坐标元组), ...]创建fileName_frame.xml
                    frame_count += 1
                else:
                    break
            cap.release()
            self.dict.clear()
        if tag == "track" and self.isPerson:
            self.isPerson = False
        if tag == "box" and self.isPerson:
            # 每隔x帧取字典dict: {frame: [(坐标元组), ...]}
            if int(self.frame) % 10 == 0:
                tup = (self.ybr, self.xbr, self.ytl, self.xtl)
                self.dict.setdefault(int(self.frame), []).append(tup)

    # 基于[(坐标元组), ...]创建fileName_frame.xml
    def createXML(self, frame_count):
        implementation = getDOMImplementation()
        document = implementation.createDocument(None, None, None)

        annotation = document.createElement("annotation")
        document.appendChild(annotation)
        folder = document.createElement("folder")
        folder.appendChild(document.createTextNode("JPEGImages"))
        annotation.appendChild(folder)
        filename = document.createElement("filename")
        filename.appendChild(document.createTextNode(fileName + "_" + str(frame_count)))
        annotation.appendChild(filename)
        path = document.createElement("path")
        path.appendChild(document.createTextNode("C:/AIoT/Projects/keras-yolo3-master/VOCdevkit/VOC2007/JPEGImages/" + fileName + "_%d.jpg" % frame_count))
        annotation.appendChild(path)
        source = document.createElement("source")
        annotation.appendChild(source)
        database = document.createElement("database")
        database.appendChild(document.createTextNode("Unknown"))
        source.appendChild(database)
        size = document.createElement("size")
        annotation.appendChild(size)
        width = document.createElement("width")
        width.appendChild(document.createTextNode("1920"))
        size.appendChild(width)
        height = document.createElement("height")
        height.appendChild(document.createTextNode("1080"))
        size.appendChild(height)
        depth = document.createElement("depth")
        depth.appendChild(document.createTextNode("3"))
        size.appendChild(depth)
        segmented = document.createElement("segmented")
        segmented.appendChild(document.createTextNode("0"))
        annotation.appendChild(segmented)

        tup_list = self.dict.get(frame_count)
        for tup in tup_list:
            object = document.createElement("object")
            annotation.appendChild(object)
            name = document.createElement("name")
            name.appendChild(document.createTextNode("person"))
            object.appendChild(name)
            pose = document.createElement("pose")
            pose.appendChild(document.createTextNode("Unspecified"))
            object.appendChild(pose)
            truncated = document.createElement("truncated")
            truncated.appendChild(document.createTextNode("0"))
            object.appendChild(truncated)
            difficult = document.createElement("difficult")
            difficult.appendChild(document.createTextNode("0"))
            object.appendChild(difficult)
            bndbox = document.createElement("bndbox")
            object.appendChild(bndbox)
            xmin = document.createElement("xmin")
            xmin.appendChild(document.createTextNode(tup[3]))
            bndbox.appendChild(xmin)
            ymin = document.createElement("ymin")
            ymin.appendChild(document.createTextNode(tup[2]))
            bndbox.appendChild(ymin)
            xmax = document.createElement("xmax")
            xmax.appendChild(document.createTextNode(tup[1]))
            bndbox.appendChild(xmax)
            ymax = document.createElement("ymax")
            ymax.appendChild(document.createTextNode(tup[0]))
            bndbox.appendChild(ymax)

        out = open(xml_save_path + "/" + fileName + "_%d.xml" % frame_count, "w")
        document.writexml(out, '', '\t', '\n')
        print("Save xml: " + fileName + "_%d.xml" % frame_count)


if (__name__ == "__main__"):
    # 创建一个XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    # 重写 ContextHandler
    Handler = DataHandler()
    parser.setContentHandler(Handler)

    xml_list = os.listdir(xml_src_path)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for each_xml in xml_list:
        fileName, _ = each_xml.split('.')
        parser.parse(xml_src_path + "/" + each_xml)
