#import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import lxml
import os
import argparse
import sys
from xml.etree.ElementTree import parse
E = ET.Element


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="交通信号灯逻辑输出文件",
        epilog="python run.py file_name (.xml将自动添加)")

    # optional input parameters
    parser.add_argument(
        '--file', type=str, default=None,
        help='比较组文件名')
    return parser.parse_known_args(args)[0]


def tl_logic(file_path):
    ids=str()
    # 加载文件
    tree = parse(file_path)
    # 设置文件写入
    tl_additional_default = ET.Element('additional')
    tl_additional = ET.Element('additional')

    # 查找并写入
    trafficSignalList = tree.findall('trafficSignal')
    # 在trafficSignal内查找
    for trafficSignal in trafficSignalList:
        TODPlan = trafficSignal.findall('TODPlan')
        scheduleList = trafficSignal.findall('schedule')
        # 在这里写下你想找的id和start time
        for plan in TODPlan[0].findall('plan'):
            if plan.attrib['startTime'] == '9000':  # 9点 25200 # 2点半? 9000
                ids = plan.attrib['schedule']

        # 在schedule内查找的过程
        for schedule in scheduleList:
            # 更换node id
            if TODPlan[0].attrib['defaultPlan'] == schedule.attrib['id']:  # default
                schedule.attrib['programID']=schedule.attrib['id']
                schedule.attrib['id'] = trafficSignal.attrib['nodeID']
                schedule.attrib['type']='static'
                tlLogic_default = ET.SubElement(
                    tl_additional_default, 'tlLogic', attrib=schedule)
                phase_set = schedule.findall('phase')
                for phase in phase_set:
                    phase_dict = dict()
                    for key in phase.keys():
                        phase_dict[key] = phase.attrib[key]
                    tlLogic_default.append(E('phase', attrib=phase_dict))

            if ids == schedule.attrib['id']:
                schedule.attrib['programID']=schedule.attrib['id']
                schedule.attrib['id'] = trafficSignal.attrib['nodeID']
                schedule.attrib['type']='static'
                tlLogic = ET.SubElement(
                    tl_additional, 'tlLogic', attrib=schedule)
                phase_set = schedule.findall('phase')
                for phase in phase_set:
                    phase_dict = dict()
                    for key in phase.keys():
                        phase_dict[key] = phase.attrib[key]
                    tlLogic.append(E('phase', attrib=phase_dict))

    print("end")

    dump(tl_additional_default)
    writetree = ET.ElementTree(tl_additional_default)
    writetree.write(os.path.join('./output_default_tl.add.xml'),
                    pretty_print=True, encoding='UTF-8', xml_declaration=True)
    writetree = ET.ElementTree(tl_additional)
    writetree.write(os.path.join('./output_tl.add.xml'),
                    pretty_print=True, encoding='UTF-8', xml_declaration=True)
    print(scheduleList)

    # if len(self.traffic_light) != 0 or .configs['mode'] == 'simulate':
    #     for _, tl in enumerate(traffic_light_set):
    #         phase_set = tl.pop('phase')
    #         tlLogic = ET.SubElement(tl_additional, 'tlLogic', attrib=tl)
    #         indent(tl_additional, 1)
    #         for _, phase in enumerate(phase_set):
    #             tlLogic.append(E('phase', attrib=phase))
    #             indent(tl_additional, 2)

    # dump(tl_additional)
    # tree = ET.ElementTree(tl_additional)
    # tree.write(os.path.join(path, .file_name+'_tl.add.xml'),
    #             pretty_print=True, encoding='UTF-8', xml_declaration=True)


def main(args):
    flags = parse_args(args)
    path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), flags.file)

    tl_logic(path)


if __name__ == '__main__':
    main(sys.argv[1:])
