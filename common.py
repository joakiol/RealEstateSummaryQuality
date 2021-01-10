import json
from json2html import *
from collections import OrderedDict
import re
import nltk


def make_custom_sort(orders):
    orders = [{k: -i for (i, k) in enumerate(reversed(order), 1)} for order in orders]

    def process(stuff):
        if isinstance(stuff, dict):
            l = [(k, process(v)) for (k, v) in stuff.items()]
            keys = set(stuff)
            for order in orders:
                if keys.issuperset(order):
                    return OrderedDict(sorted(l, key=lambda x: order.get(x[0], 0)))
            return OrderedDict(sorted(l))
        if isinstance(stuff, list):
            return [process(x) for x in stuff]
        return stuff

    return process


class Cadastre:
    def __init__(self, Knr, Gnr, Bnr, Fnr, Snr, Anr):
        self.Knr = Knr
        self.Gnr = Gnr
        self.Bnr = Bnr
        self.Fnr = Fnr
        self.Snr = Snr
        self.Anr = Anr

    def __iter__(self):
        return self

    def __repr__(self):
        return str(self.__dict__)


class Building:
    def __init__(self, build_year, build_type, areal_bra, areal_boa, areal_prom, debt, fortune, constr_cost):
        self.build_year = build_year
        self.build_type = build_type
        self.areal_bra = areal_bra
        self.areal_boa = areal_boa
        self.areal_prom = areal_prom
        self.debt = debt
        self.fortune = fortune
        self.constr_cost = constr_cost

    def __iter__(self):
        return self

    def __repr__(self):
        return str(self.__dict__)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class PlaceDescription:
    def __init__(self, type, description):
        self.type = type
        self.description = description

    def __iter__(self):
        return self

    def __repr__(self):
        return str(self.__dict__)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class ConditionDescription:
    def __init__(self, type, room, description, assessment, degree):
        self.type = type
        self.room = room
        self.description = description
        self.assessment = assessment
        self.degree = degree
        self.adjusted = False

    def adjust_tg(new_degree):
        self.adjusted = True
        self.original_degree = self.degree
        self.degree = new_degree

    def __iter__(self):
        return self

    def __repr__(self):
        return str(self.__dict__)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class ConditionReport:
    def __init__(self, id, type, date, author, building, cadastre, place, condition):
        self.id = id
        self.type = type
        self.date = date
        self.author = author
        self.building = building
        self.cadastre = cadastre
        self.place = place
        self.condition = condition

    def __iter__(self):
        return self

    def __repr__(self):
        return str(self.__dict__)

    def getText(self):
        builder = " ".join([d.description for d in self.condition]) + " "
        builder += " ".join([p.description for p in self.place])
        return builder

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class SummaryReport(ConditionReport):
    """Extends the `ConditionReport` class, with some useful functions for this project."""
    def __init__(self, cr):
        """
        Constructor for report. 

        :param cr: ConditionReport object. A real-estate condition report.     
        """
        super(SummaryReport, self).__init__(cr.id, cr.type, cr.date, cr.author, 
                                     cr.building, cr.cadastre, cr.place, cr.condition)
        self.summary = self._find_summary()
        self._clean()

    def _find_summary(self):
        """Set summary from place number 19/20, after cleaning. """
        summary = ''
        seen = set()
        for element in self.place:
            clean_descr = self._clean_text(element.description)
            if element.type not in seen and (element.type == 19 or element.type == 20) and len(clean_descr) > 0:
                seen.add(element.type)
                summary += self._clean_text(clean_descr) + ' '
        return summary.strip()

    def _clean_text(self, text):
        """
        Cleans a string input the following way: 
        - ensure period at end. 
        - ensure no trailing spaces. 
        - ensure no double space or period. 
        """
        space_re = re.compile(r"\s+")
        dot_re = re.compile(r"\.\.+")

        text = text.replace('/', ' / ')
        text = space_re.sub(' ', text).strip()
        text = dot_re.sub('.', text)

        if text == '':
            return text
        elif text[-1] == ',':
            string_list = list(text)
            string_list[-1] = '.'
            text = ''.join(string_list)
        elif text[-1] not in ['.', '!', '?', ':']:
            text += '.'

        return text

    def _clean(self):
        """Clean condition in report. Summary was already cleaned under construction. """
        new_condition = []
        seen = set()

        for element in self.condition:
            roomtype = (element.type, element.room)
            clean_descr = self._clean_text(element.description)
            clean_assess = self._clean_text(element.assessment)

            if roomtype not in seen and (len(clean_descr) > 0 or len(clean_assess) > 0):
                seen.add(roomtype)
                element.description = clean_descr
                element.assessment = clean_assess
                new_condition.append(element)

        self.condition = new_condition
        return self

    def get_report_raw(self):
        """Return the complete report text, as a raw string."""
        text = ''
        for element in self.condition:
            if element.description == '':
                text += element.assessment + ' '
            elif element.assessment == '':
                text += element.description + ' '
            else: 
                text += element.description + ' ' + element.assessment + ' '
        return text.strip()

    def get_summary_raw(self):
        """Return the complete summary text, as a raw string."""
        return self.summary

    def get_sections(self):
        """Return the sections of the report, as a list of strings."""
        report = []
        for element in self.condition:
            if element.description == '':
                text = element.assessment
            elif element.assessment == '':
                text = element.description
            else: 
                text = element.description + ' ' + element.assessment
            report.append(text)
        return report

    def get_report_words(self):
        """Return the words of the report, as a list of strings."""
        raw = self.get_report_raw()
        report = [word.lower() for word in nltk.word_tokenize(raw, language='norwegian')]
        return report

    def get_summary_words(self):
        """Return the words of the summary, as a list of strings."""
        summary = [word.lower() for word in nltk.word_tokenize(self.summary, language='norwegian')]
        return summary

    def get_report_sentences(self):
        """Return the sentences of the report, as a list of strings."""
        raw = self.get_report_raw()
        return nltk.sent_tokenize(raw, language="norwegian")
    
    def get_summary_sentences(self):
        """Return the sentences of the summary, as a list of strings."""
        return nltk.sent_tokenize(self.summary, language='norwegian')

    def get_tokenized_sections(self):
        """Return the tokenized sections of the report, as a list of {list of strings}."""
        sections = self.get_sections()
        report = [[word.lower() for word in nltk.word_tokenize(section, language='norwegian')] 
                  for section in sections]
        return report

    def get_report_tokenized_sentences(self):
        """Return the tokenized sentences of the report, as a list of {list of strings}."""
        sentences = self.get_report_sentences()
        report = [[word.lower() for word in nltk.word_tokenize(sentence, language='norwegian')] 
                  for sentence in sentences]
        return report

    def get_summary_tokenized_sentences(self):
        """Return the tokenized sentences of the summary, as a list of {list of strings}."""
        sentences = self.get_summary_sentences()
        summary = [[word.lower() for word in nltk.word_tokenize(sentence, language='norwegian')] 
                  for sentence in sentences]
        return summary


def get_report_html(report):
    frozen = report.toJSON()
    return json2html.convert(json=frozen)