import logging
import os
import tempfile
from typing import List, Dict
import xml.dom.minidom
# from xml.dom.minidom import Element
from xml.etree.ElementTree import ElementTree, Element
from xml.etree.ElementTree import tostring

from pan import main as pan_metric

class IntrinsicCase:
    def __init__(self, cases: List, document_name: str):
        self.file_name = document_name
        self.cases_index = cases


class PanIntrinsicMetricGlobal:
    def calculate(self, reals: List[IntrinsicCase], preds: List[IntrinsicCase]) -> dict:

        xmls_real = dict()
        xmls_pred = dict()
        for case in reals:
            try:
                xmls_real[case.file_name] = self._get_xml(self._get_json(case), case.file_name)
            except Exception as e:
                logging.error(str(e))

        for case in preds:
            try:
                xmls_pred[case.file_name] = self._get_xml(self._get_json(case), case.file_name, name='detected')
            except Exception as e:
                logging.error(str(e))

        logging.debug(f'xmls_real={len(xmls_real)}, xmls_pred={len(xmls_pred)}')

        dirpath_real = tempfile.mkdtemp()
        dirpath_pred = tempfile.mkdtemp()

        for file in xmls_pred:
            with open(os.path.join(dirpath_pred, f"{file.split('.')[0]}.xml"), 'w') as f:
                f.write(xmls_pred[file])
            with open(os.path.join(dirpath_real, f"{file.split('.')[0]}.xml"), 'w') as f:
                f.write(xmls_real[file])

        metrics = pan_metric(False, dirpath_real, "plagiarised", dirpath_pred, "detected")

        return metrics

    def _get_json(self, case: IntrinsicCase) -> List[Dict[str, int]]:
        blocks = []

        for span in case.cases_index:
            blocks.append({'Offset': span[0], 'Length': span[1] - span[0]})
        return blocks

    def _get_xml(self, sources: List[Dict[str, int]], filename: str, name: str = 'plagiarised') -> str:
        markup = []
        for elemen in sources:
            markup.append(
                Element('feature',
                        name=filename,
                        this_offset=str(elemen['Offset']),
                        this_length=str(elemen['Length']),
                        source_reference=str(elemen['Author']),
                        source_offest=str(elemen['Offset']),
                        source_length=str(elemen['Length'])))

        document = Element('document', reference=filename)
        document.extend(markup)
        rough_string = tostring(document, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
