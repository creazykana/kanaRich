# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

keys = ["period_count_r3", "count_price_level1_r1", "amt_price_level1_r1", "count_profit_rate_level4_r1",
            "qty_profit_rate_level4_r1"]
# closing down model variables 39.
def scoreCards2(r):
    score = 547.4192
    varDict = {
        "period_count_r3": [[12, 13, 14, np.inf], [-39.9601, -3.4866, 21.6548, 37.6396]],
        "count_price_level1_r1": [[4, 7, 10, 16, np.inf], [-25.0438, -13.6447, -5.056, 4.0692, 21.2147]],
        "amt_price_level1_r1": [[790, 1490, 2390, 5090, 8890, np.inf],
                                [-18.5974, -10.1026, 0.8933, 8.4578, 14.1215, 21.4092]],
        "count_profit_rate_level4_r1": [[3, 6, np.inf], [-21.7486, -3.3511, 19.3341]],
        "qty_profit_rate_level4_r1": [[0.056, 0.132, 0.24, 0.324, 0.5, np.inf],
                                      [-12.1334, -7.913, -2.6192, 1.9911, 5.0727, 10.2427]]}
    keys = ["period_count_r3", "count_price_level1_r1", "amt_price_level1_r1", "count_profit_rate_level4_r1",
            "qty_profit_rate_level4_r1"]
    for key, i in zip(keys, range(len(keys))):
        segBounds = varDict[key][0]
        segScores = varDict[key][1]
        for seg, segScore in zip(segBounds, segScores):
            if r[i] < seg:
                score = score + segScore
                break
    return score


def period_count_r3_map(r):
    if r > -9999999 and r < 12:
        return -39.9601
    elif r >= 12 and r < 13:
        return -3.4866
    elif r >= 13 and r < 14:
        return 21.6548
    elif r >= 14:
        return 37.6396


def count_price_level1_r1_map(r):
    if r > -9999999 and r < 4:
        return -25.0438
    elif r >= 4 and r < 7:
        return -13.6447
    elif r >= 7 and r < 10:
        return -5.056
    elif r >= 10 and r <= 16:
        return 4.0692
    elif r >= 16:
        return 21.2147


def amt_price_level1_r1_map(r):
    if r > -9999999 and r < 790:
        return -18.5974
    elif r >= 790 and r < 1490:
        return -10.1026
    elif r >= 1490 and r < 2390:
        return 0.8933
    elif r >= 2390 and r < 5090:
        return 8.4578
    elif r >= 5090 and r < 8890:
        return 14.1215
    elif r >= 8890:
        return 21.4092


def count_profit_rate_level4_r1_map(r):
    if r > -9999999 and r < 3:
        return -21.7486
    elif r >= 3 and r < 6:
        return -3.3511
    elif r >= 6:
        return 19.3341


def qty_profit_rate_level4_r1_map(r):
    if r > -9999999 and r < 0.056:
        return -12.1334
    elif r >= 0.056 and r < 0.132:
        return -7.913
    elif r >= 0.132 and r < 0.24:
        return -2.6192
    elif r >= 0.24 and r < 0.324:
        return 1.9911
    elif r >= 0.324 and r < 0.5:
        return 5.0727
    elif r >= 0.5:
        return 10.2427


def parseRow(row):
    cust_code = row[0]
    # period_count_r3 = row[1]
    # count_price_level1_r1 = row[2]
    # amt_price_level1_r1 = row[3]
    # count_profit_rate_level4_r1 = row[4]
    # qty_profit_rate_level4_r1 = row[5]
    result = [cust_code]
    for i in range(len(keys)):
        var = keys[i]
        r = row[i+1]
        result.append(eval(var+"_map")(r))
    basic_score = 547.4192
    score = np.sum(result[1:]) + basic_score
    result.append(score)
    return result


def getMapFunc():
    varDict = {
        "period_count_r3": [[12, 13, 14, np.inf], [-39.9601, -3.4866, 21.6548, 37.6396]],
        "count_price_level1_r1": [[4, 7, 10, 16, np.inf], [-25.0438, -13.6447, -5.056, 4.0692, 21.2147]],
        "amt_price_level1_r1": [[790, 1490, 2390, 5090, 8890, np.inf],
                                [-18.5974, -10.1026, 0.8933, 8.4578, 14.1215, 21.4092]],
        "count_profit_rate_level4_r1": [[3, 6, np.inf], [-21.7486, -3.3511, 19.3341]],
        "qty_profit_rate_level4_r1": [[0.056, 0.132, 0.24, 0.324, 0.5, np.inf],
                                      [-12.1334, -7.913, -2.6192, 1.9911, 5.0727, 10.2427]]}
    space = "    "
    for key, values in varDict.items():
        print("def %s_map(r):" % key)
        for i in range(len(values[0])):
            if i == 0:
                print(space + "if r>-9999999 and r<=%s:" % str(values[0][i]))
                print(space * 2 + "return %s" % values[1][i])
            else:
                print(space + "elif r>%d and r<=%s:" % (values[0][i - 1], values[0][i]))
                print(space * 2 + "return %s" % values[1][i])


if __name__ == "__main__":
    statTimeRange = [["201707", "201709"], ["201802", "201804"], ["201807", "201809"], ["201902", "201904"]]
    for timeRange in statTimeRange:
        startMonth = timeRange[0]
        endMonth = timeRange[1]
        print(">>>%s time range data is down." % (startMonth + "_" + endMonth))