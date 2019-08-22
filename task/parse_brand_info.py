# -*- coding: utf-8 -*-
import pandas as pd




def reBrandNames(string):
    result = re.sub(u"（", u"_", string)
    result = re.sub(u"）", u"_", result)
    result = re.sub(u"\(", u"_", result)
    result = re.sub(u"\)", u"_", result)
    result = re.sub(u"\+", u"plus", result)
    result = re.sub(u"\.", u"point", result)
    result = re.sub(u"·", u"point", result)
    result = re.sub(u"㎎", u"mg", result)
    result = re.sub(u'[^\u4e00-\u9fa5|\w]+', u'', result)
    return result