# -*- coding: utf-8 -*-
"""
Customer Intelligence Engine — Financial Health, Engagement & Next Best Action
Team FT-US (Leader: Trần Long Triều) · Home Credit · BI Season 10 · Round 2 (Synthesize)

Run:   pip install -r requirements.txt
       python -m streamlit run app.py

Light-theme hard lock
---------------------
Every Plotly figure goes through apply_pro_theme() and is rendered with st.plotly_chart(..., theme=None),
so Streamlit can never repaint axis labels, legends or tick fonts with its dark template. All widget
chrome (select boxes, popovers, list boxes, text / number inputs, multi-selects, sliders, tables) is
pinned to a white surface with navy ink by the CSS block below. The dashboard therefore looks identical
whether the operating system / browser is in Light Mode or Dark Mode.

Data:  the app looks next to itself (or in ./data) for, in this order:
       customer_scores_nba.parquet / .csv / .csv.gz → customer_scores_sample.parquet / .csv / .csv.gz
       A file can also be uploaded from the sidebar. With no file at all it builds a 10,000-row synthetic
       demo set, so the app always opens. Portfolio-level charts use the official full-base numbers
       hard-coded in OFFICIAL below (exported from summary_metrics.json of the final run).
"""
import io
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# =============================================================================================================
# Official full-base results (1,000,528 customers, scored as of 4 Jun 2025) - exported from summary_metrics.json
# =============================================================================================================
OFFICIAL = {'tradeoff_loans': {80: {'approved': 0.1683, 'bad_rate': 0.00539, 'avoided': 0.9378},
                    70: {'approved': 0.2652, 'bad_rate': 0.00546, 'avoided': 0.9009},
                    65: {'approved': 0.3643, 'bad_rate': 0.00616, 'avoided': 0.8463},
                    60: {'approved': 0.4766, 'bad_rate': 0.00708, 'avoided': 0.7689},
                    55: {'approved': 0.7035, 'bad_rate': 0.01053, 'avoided': 0.4924},
                    50: {'approved': 0.8134, 'bad_rate': 0.01144, 'avoided': 0.3626},
                    40: {'approved': 0.9659, 'bad_rate': 0.01352, 'avoided': 0.1056},
                    30: {'approved': 0.9844, 'bad_rate': 0.01389, 'avoided': 0.0635},
                    0: {'approved': 1.0, 'bad_rate': 0.0146, 'avoided': 0.0}},
 'tradeoff_vnd': {0: {'approved': 1.0, 'bad_rate': 0.01815, 'avoided': 0.0},
                  40: {'approved': 0.9679, 'bad_rate': 0.01702, 'avoided': 0.0924},
                  55: {'approved': 0.7443, 'bad_rate': 0.01332, 'avoided': 0.4538},
                  60: {'approved': 0.565, 'bad_rate': 0.01078, 'avoided': 0.6645},
                  80: {'approved': 0.2449, 'bad_rate': 0.00927, 'avoided': 0.875}},
 'iv': {'income': 0.0035,
        'dti_existing': 0.0039,
        'n_active_inst': 0.0043,
        'AGE_AT_APP': 0.0095,
        'scoring_channels_90': 0.0163,
        'NAME_GENDER': 0.0178,
        'PTI': 0.0233,
        'rej_365': 0.0256,
        'apps_90': 0.0266,
        'FLAG_INS': 0.0397,
        'prior_severe': 0.0431,
        'PAYMENT_NUM': 0.0504,
        'DOWNPAY_RATIO': 0.0589,
        'tenure_days': 0.0722,
        'NAME_EDUCATION_TYPE': 0.0987,
        'prior_booked': 0.149,
        'max_offer': 0.1989,
        'prior_bad': 0.2108},
 'scorecard': {'Past late payments': {'0': 2.5474, '1': -80.5037, '2+': -87.2162},
               'Past write-off / sold debt': {'0': 0.1483, '1+': -38.2475},
               'Loans successfully taken (track record)': {'0': -10.207, '1': -10.207, '2': -4.9296, '3-4': 3.1702, '5+': 23.6468},
               'Relationship tenure': {'1-3y': -1.1123, '3-5y': 2.7339, '5y+': 2.7339, '<1y': -3.6873, 'No history': -3.118},
               'Rejections last 12m': {'0': 1.1034, '1': -7.8183, '2+': -19.7982},
               'Applications last 90d': {'0': 2.0128, '1': -5.4738, '2': -8.9906, '3+': -36.7832},
               'Channels shopped last 90d (credit hunger)': {'0': 2.5948, '1': -3.0979, '2': -11.2279, '3+': -19.8972},
               'Declared monthly income': {'12-20M': 0.0457, '5-8M': -0.2183, '8-12M': 0.0457, '<5M': -0.2183, '>20M': 0.0457}},
 'scorecard_badrate': {'Past late payments': {'0': 0.00462, '1': 0.06147, '2+': 0.18919},
                       'Past write-off / sold debt': {'0': 0.00495, '1+': 0.18129},
                       'Loans successfully taken (track record)': {'0': 0.00678, '1': 0.00694, '2': 0.00583, '3-4': 0.00454, '5+': 0.0024},
                       'Relationship tenure': {'1-3y': 0.00558, '3-5y': 0.00358, '5y+': 0.00395, '<1y': 0.00717, 'No history': 0.00678},
                       'Rejections last 12m': {'0': 0.00479, '1': 0.00688, '2+': 0.01125},
                       'Applications last 90d': {'0': 0.00471, '1': 0.00592, '2': 0.00662, '3+': 0.01598},
                       'Channels shopped last 90d (credit hunger)': {'0': 0.00463, '1': 0.00549, '2': 0.007, '3+': 0.00926},
                       'Declared monthly income': {'12-20M': 0.00483, '5-8M': 0.00564, '8-12M': 0.00486, '<5M': 0.00547, '>20M': 0.00541}},
 'tier_counts': {'Resilient': 321237, 'Stable': 205727, 'Vulnerable': 229195, 'Distressed': 61949, 'Thin-file': 182420},
 'tier_bad_rate': {'Distressed': 0.044, 'Vulnerable': 0.0148, 'Stable': 0.0097, 'Resilient': 0.0056, 'Thin-file': 0.0246},
 'tier_median_offer': {'Distressed': 2000000.0, 'Vulnerable': 2000000.0, 'Stable': 5000000.0, 'Resilient': 12000000.0},
 'drivers_by_tier': {'Distressed': {'Stress signal': 0.91, 'Short history': 0.09, 'Other': 0.0},
                     'Vulnerable': {'Stress signal': 0.1272, 'Short history': 0.8728, 'Other': 0.0},
                     'Stable': {'Stress signal': 0.3205, 'Short history': 0.6795, 'Other': 0.0},
                     'Resilient': {'Stress signal': 0.2824, 'Short history': 0.0976, 'Other': 0.6199}},
 'stressed': {'distressed': 61949,
              'vulnerable_with_stress_signal': 29153,
              'stressed_total': 91102,
              'stressed_share': 0.09105392352837702,
              'vulnerable_short_history_clean': 200042,
              'stressed_or_live_alarm': 104746},
 'subgroups': {'Resilient': 321237,
               'Stable': 205727,
               'Vulnerable - building history': 200042,
               'Thin-file': 182420,
               'Distressed': 61949,
               'Vulnerable - stress signal': 29153},
 'benchmark': {'FHS scorecard (8 behavioural factors, explainable)': 0.661,
               'Gradient boosting (black box, +MAX_OFFER, loan & demographic vars)': 0.6964,
               'Demographics only (age, gender, education)': 0.616,
               'Income only': 0.5074},
 'pillar_weights': {'Repayment discipline': 67.14, 'Credit-seeking stress': 32.76, 'Financial capacity': 0.11},
 'es_level_mix': {'Cooling': 0.3846, 'Warm': 0.3156, 'Active': 0.2002, 'Champion': 0.0995},
 'es_cuts': [24.0, 38.0, 54.0],
 'es_profile': {'Cooling': {'days_since_last_cie': 155.074,
                            'cie_12m': 1.242,
                            'gma_sessions': 0.005,
                            'ob_attempts_12m': 0.364,
                            'ob_connected_12m': 0.03,
                            'inbound_12m': 0.001,
                            'inbound_clean_12m': 0.0,
                            'ob_positive_12m': 0.0,
                            'pull_leads_decayed': 0.003,
                            'n_push_leads': 0.9,
                            'families_12m': 1.02,
                            'holds_active_product': 0.253,
                            'R': 30.614,
                            'F': 15.989,
                            'I': 0.09,
                            'L': 13.149,
                            'P': 24.261,
                            'ES': 18.312,
                            'connect_rate': 0.082,
                            'n': 384846.0},
                'Warm': {'days_since_last_cie': 136.697,
                         'cie_12m': 2.255,
                         'gma_sessions': 0.048,
                         'ob_attempts_12m': 0.939,
                         'ob_connected_12m': 0.071,
                         'inbound_12m': 0.011,
                         'inbound_clean_12m': 0.01,
                         'ob_positive_12m': 0.002,
                         'pull_leads_decayed': 0.039,
                         'n_push_leads': 1.695,
                         'families_12m': 1.468,
                         'holds_active_product': 0.771,
                         'R': 36.373,
                         'F': 32.002,
                         'I': 0.85,
                         'L': 14.202,
                         'P': 45.052,
                         'ES': 29.686,
                         'connect_rate': 0.076,
                         'n': 315811.0},
                'Active': {'days_since_last_cie': 85.23,
                           'cie_12m': 4.337,
                           'gma_sessions': 0.093,
                           'ob_attempts_12m': 1.496,
                           'ob_connected_12m': 0.117,
                           'inbound_12m': 0.062,
                           'inbound_clean_12m': 0.057,
                           'ob_positive_12m': 0.014,
                           'pull_leads_decayed': 0.063,
                           'n_push_leads': 2.092,
                           'families_12m': 2.32,
                           'holds_active_product': 0.869,
                           'R': 56.485,
                           'F': 48.04,
                           'I': 2.071,
                           'L': 14.579,
                           'P': 69.01,
                           'ES': 44.73,
                           'connect_rate': 0.079,
                           'n': 200323.0},
                'Champion': {'days_since_last_cie': 27.082,
                             'cie_12m': 11.947,
                             'gma_sessions': 0.252,
                             'ob_attempts_12m': 2.038,
                             'ob_connected_12m': 0.149,
                             'inbound_12m': 0.152,
                             'inbound_clean_12m': 0.139,
                             'ob_positive_12m': 0.037,
                             'pull_leads_decayed': 0.169,
                             'n_push_leads': 1.692,
                             'families_12m': 3.122,
                             'holds_active_product': 0.951,
                             'R': 82.729,
                             'F': 72.664,
                             'I': 4.437,
                             'L': 18.999,
                             'P': 86.739,
                             'ES': 62.595,
                             'connect_rate': 0.073,
                             'n': 99548.0}},
 'es_forward': {'Cooling': {'connect_per_call': 0.0837,
                            'positive_per_100_calls': 0.2978,
                            'wins_per_1000_calls': 0.2852,
                            'customers_called': 62089.0,
                            'inbound_rate': 0.0162},
                'Warm': {'connect_per_call': 0.0768,
                         'positive_per_100_calls': 0.4062,
                         'wins_per_1000_calls': 0.4413,
                         'customers_called': 118649.0,
                         'inbound_rate': 0.0408},
                'Active': {'connect_per_call': 0.077,
                           'positive_per_100_calls': 0.8113,
                           'wins_per_1000_calls': 1.1228,
                           'customers_called': 105368.0,
                           'inbound_rate': 0.0778},
                'Champion': {'connect_per_call': 0.0691,
                             'positive_per_100_calls': 1.376,
                             'wins_per_1000_calls': 1.7183,
                             'customers_called': 55285.0,
                             'inbound_rate': 0.1243}},
 'es_backtest_active': {'Dormant': 0.1843, 'Cooling': 0.209, 'Warm': 0.165, 'Active': 0.3122, 'Champion': 0.5931},
 'es_validation': {'Back-test: stays engaged (self-initiated activity 2025H1)': {'n': 821623, 'rate': 0.2478, 'auc': 0.6188},
                   'Forward: positive / win response (≥1 sales call)': {'n': 341391, 'rate': 0.0321, 'auc': 0.6633},
                   'Forward HOLD-OUT half (not used for weights): positive response': {'n': 170368, 'rate': 0.0319, 'auc': 0.6622},
                   'Forward: same, ES without GMA (leakage check)': {'n': 341391, 'rate': 0.0321, 'auc': 0.6703},
                   'Forward: customer calls us inbound': {'n': 1000528, 'rate': 0.047, 'auc': 0.7162},
                   'Forward: answers ≥1 sales call (reachability, not engagement)': {'n': 341391, 'rate': 0.4341, 'auc': 0.5179}},
 'es_drivers': [{'factor': '− Self-initiated activity in last 90d',
                 'share': 0.6709,
                 'lift': 0.899,
                 'outcome': 'stays self-active (back-test)'},
                {'factor': '+ Self-generated lead (GMA/landing/SOB)',
                 'share': 0.0375,
                 'lift': 1.856,
                 'outcome': 'stays self-active (back-test)'},
                {'factor': '+ Explored ≥2 product families', 'share': 0.3464, 'lift': 2.44, 'outcome': 'stays self-active (back-test)'},
                {'factor': '+ Holds an active product', 'share': 0.5098, 'lift': 1.291, 'outcome': 'stays self-active (back-test)'},
                {'factor': '− Only pushed leads & said "Not interested" ≥2×',
                 'share': 0.1205,
                 'lift': 0.647,
                 'outcome': 'stays self-active (back-test)'},
                {'factor': '− Rejected in last 90d', 'share': 0.176, 'lift': 0.845, 'outcome': 'stays self-active (back-test)'},
                {'factor': '+ A loan matures in next 90 days',
                 'share': 0.0945,
                 'lift': 1.377,
                 'outcome': 'applies for a new loan (back-test)'},
                {'factor': '− Just booked a loan (≤90d)', 'share': 0.4072, 'lift': 0.374, 'outcome': 'applies for a new loan (back-test)'},
                {'factor': '○ Uses the mobile app (GMA)',
                 'share': 0.0935,
                 'lift': 0.992,
                 'outcome': 'positive outcome per sales call (forward)'},
                {'factor': '+ Called us inbound', 'share': 0.0578, 'lift': 3.634, 'outcome': 'positive outcome per sales call (forward)'},
                {'factor': '+ Responded positively before',
                 'share': 0.0155,
                 'lift': 20.321,
                 'outcome': 'positive outcome per sales call (forward)'},
                {'factor': '− ≥5 unanswered sales calls (call fatigue)',
                 'share': 0.1255,
                 'lift': 0.814,
                 'outcome': 'positive outcome per sales call (forward)'},
                {'factor': '+ Picked up and refused before (a "no" is still contact)',
                 'share': 0.1058,
                 'lift': 2.718,
                 'outcome': 'positive outcome per sales call (forward)'}],
 'es_weights': {'R': {'expert': 0.2, 'data': 0.2532, 'used': 0.2},
                'F': {'expert': 0.3, 'data': 0.2451, 'used': 0.3},
                'I': {'expert': 0.15, 'data': 0.2599, 'used': 0.15},
                'L': {'expert': 0.1, 'data': 0.0902, 'used': 0.1},
                'P': {'expert': 0.25, 'data': 0.1515, 'used': 0.25}},
 'es_weight_holdout': {'expert': 0.6622,
                       'data_driven': 0.6759,
                       'backtest_expert': 0.6188,
                       'backtest_data_driven': 0.6024,
                       'rule': 'data-driven weights only if not worse on the forward hold-out AND the back-test',
                       'used': 'expert'},
 'refractory': {'(0, 30]': 0.2175, '(30, 90]': 0.3849, '(90, 180]': 0.3743, '(180, 365]': 0.2916, '(365, 730]': 0.2528},
 'speed_to_lead': {'<1h': 0.2653, '1-4h': 0.2048, '4-24h': 0.2267, '1-3d': 0.2333, '>3d': 0.506},
 'es_decile': {'stays self-active (back-test)': {1: 0.1828,
                                                 2: 0.2375,
                                                 3: 0.2546,
                                                 4: 0.1247,
                                                 5: 0.1968,
                                                 6: 0.1314,
                                                 7: 0.1811,
                                                 8: 0.2653,
                                                 9: 0.3326,
                                                 10: 0.5713},
               'positive per 100 sales calls (forward)': {1: 0.3427,
                                                          2: 0.3164,
                                                          3: 0.292,
                                                          4: 0.2813,
                                                          5: 0.3161,
                                                          6: 0.3871,
                                                          7: 0.4918,
                                                          8: 0.6089,
                                                          9: 0.971,
                                                          10: 1.372},
               'calls us inbound (forward)': {1: 0.0064,
                                              2: 0.0132,
                                              3: 0.0219,
                                              4: 0.0252,
                                              5: 0.0312,
                                              6: 0.0388,
                                              7: 0.0541,
                                              8: 0.0639,
                                              9: 0.0915,
                                              10: 0.1243}},
 'i1': {'PULL': {'leads': 71554.0, 'book_30d': 0.4351, 'apply_30d': 0.5924, 'lead_share': 0.0346, 'bookings_share': 0.0966},
        'PUSH': {'leads': 1948703.0, 'book_30d': 0.1454, 'apply_30d': 0.1788, 'lead_share': 0.9426, 'bookings_share': 0.8797},
        'WARM': {'leads': 47193.0, 'book_30d': 0.1617, 'apply_30d': 0.2204, 'lead_share': 0.0228, 'bookings_share': 0.0237}},
 'i1_source': {'SOB': {'book_30d': 0.6785, 'lead_share': 0.0181, 'initiator': 'PULL'},
               'RTDM': {'book_30d': 0.6199, 'lead_share': 0.0026, 'initiator': 'WARM'},
               'GMA': {'book_30d': 0.1828, 'lead_share': 0.0119, 'initiator': 'PULL'},
               'SA': {'book_30d': 0.169, 'lead_share': 0.4087, 'initiator': 'PUSH'},
               'LANDING PAGE': {'book_30d': 0.1331, 'lead_share': 0.0046, 'initiator': 'PULL'},
               'RA': {'book_30d': 0.1274, 'lead_share': 0.5339, 'initiator': 'PUSH'},
               'TLS': {'book_30d': 0.1031, 'lead_share': 0.0202, 'initiator': 'WARM'}},
 'i2': {'0': {'applications': 1536073.0, 'reject_rate': 0.25115, 'loans': 929527.0, 'bad_rate': 0.00927},
        '1': {'applications': 616530.0, 'reject_rate': 0.21176, 'loans': 354371.0, 'bad_rate': 0.01354},
        '2': {'applications': 154520.0, 'reject_rate': 0.26798, 'loans': 76389.0, 'bad_rate': 0.0157},
        '3': {'applications': 34490.0, 'reject_rate': 0.40325, 'loans': 12005.0, 'bad_rate': 0.02107},
        '4+': {'applications': 6641.0, 'reject_rate': 0.50745, 'loans': 1586.0, 'bad_rate': 0.03342}},
 'i3': {'matured >1y ago': {'n': 197116, 'rate': 0.5479},
        'matured 91-365d ago': {'n': 82998, 'rate': 0.6249},
        'matured ≤90d ago': {'n': 57500, 'rate': 0.6301},
        'matures in 0-30d': {'n': 33661, 'rate': 0.5949},
        'matures in 31-90d': {'n': 44008, 'rate': 0.5738},
        'matures in 91-180d': {'n': 165799, 'rate': 0.2256},
        'matures in 181-365d': {'n': 97726, 'rate': 0.2219},
        'never had instalment loan': {'n': 321720, 'rate': 0.3849}},
 'i4': {1: {'attempts': 350042, 'connect': 0.1821, 'positive': 0.00621},
        2: {'attempts': 279381, 'connect': 0.1166, 'positive': 0.00545},
        3: {'attempts': 241459, 'connect': 0.0845, 'positive': 0.00432},
        4: {'attempts': 215759, 'connect': 0.0657, 'positive': 0.00362},
        5: {'attempts': 195163, 'connect': 0.0541, 'positive': 0.00309},
        6: {'attempts': 178091, 'connect': 0.0463, 'positive': 0.00305},
        7: {'attempts': 162089, 'connect': 0.0395, 'positive': 0.00284},
        8: {'attempts': 145620, 'connect': 0.035, 'positive': 0.00266},
        9: {'attempts': 129037, 'connect': 0.0309, 'positive': 0.00289},
        10: {'attempts': 113729, 'connect': 0.0272, 'positive': 0.00294},
        11: {'attempts': 99835, 'connect': 0.024, 'positive': 0.0027},
        12: {'attempts': 86312, 'connect': 0.0219, 'positive': 0.00297},
        13: {'attempts': 72667, 'connect': 0.0205, 'positive': 0.00307},
        14: {'attempts': 59748, 'connect': 0.0188, 'positive': 0.0029},
        15: {'attempts': 485446, 'connect': 0.0112, 'positive': 0.00466}},
 'segments': {'GROW': {'customers': 510845,
                       'share': 0.5106,
                       'avg_FHS': 70.9502,
                       'avg_ES': 39.0769,
                       'bt_newloan_bad': 0.0118,
                       'bt_active': 0.2535,
                       'fwd_positive_per_100_calls': 0.7815,
                       'median_max_offer': 7000000.0},
              'REACTIVATE': {'customers': 129103,
                             'share': 0.129,
                             'avg_FHS': 72.7916,
                             'avg_ES': 20.081,
                             'bt_newloan_bad': 0.0068,
                             'bt_active': 0.2388,
                             'fwd_positive_per_100_calls': 0.2919,
                             'median_max_offer': 10000000.0},
              'PROTECT': {'customers': 210792,
                          'share': 0.2107,
                          'avg_FHS': 38.0948,
                          'avg_ES': 30.1078,
                          'bt_newloan_bad': 0.0212,
                          'bt_active': 0.2887,
                          'fwd_positive_per_100_calls': 0.679,
                          'median_max_offer': 2000000.0},
              'BUILD': {'customers': 149788,
                        'share': 0.1497,
                        'avg_FHS': 47.42,
                        'avg_ES': 18.1127,
                        'bt_newloan_bad': 0.0481,
                        'bt_active': 0.1843,
                        'fwd_positive_per_100_calls': 0.6864,
                        'median_max_offer': 11199000.0}},
 'sub_segments': {'Prime Champions': {'segment': 'GROW',
                                      'customers': 208694,
                                      'share': 0.2086,
                                      'avg_FHS': 76.0855,
                                      'avg_ES': 49.7963,
                                      'bt_newloan_bad': 0.0121,
                                      'bt_active': 0.3925,
                                      'fwd_positive_per_100_calls': 1.0428,
                                      'median_max_offer': 10000000.0,
                                      'fwd_collections': 0.1377},
                  'Loyal Core': {'segment': 'GROW',
                                 'customers': 189167,
                                 'share': 0.1891,
                                 'avg_FHS': 76.4181,
                                 'avg_ES': 30.3355,
                                 'bt_newloan_bad': 0.0088,
                                 'bt_active': 0.1791,
                                 'fwd_positive_per_100_calls': 0.398,
                                 'median_max_offer': 9000000.0,
                                 'fwd_collections': 0.076},
                  'Sleeping Prime': {'segment': 'REACTIVATE',
                                     'customers': 129103,
                                     'share': 0.129,
                                     'avg_FHS': 72.7916,
                                     'avg_ES': 20.081,
                                     'bt_newloan_bad': 0.0068,
                                     'bt_active': 0.2388,
                                     'fwd_positive_per_100_calls': 0.2919,
                                     'median_max_offer': 10000000.0,
                                     'fwd_collections': 0.0667},
                  'Cautious Growers': {'segment': 'GROW',
                                       'customers': 112984,
                                       'share': 0.1129,
                                       'avg_FHS': 52.31,
                                       'avg_ES': 33.9124,
                                       'bt_newloan_bad': 0.0196,
                                       'bt_active': 0.1626,
                                       'fwd_positive_per_100_calls': 0.6371,
                                       'median_max_offer': 2000000.0,
                                       'fwd_collections': 0.0777},
                  'Stretched Seekers': {'segment': 'PROTECT',
                                        'customers': 25089,
                                        'share': 0.0251,
                                        'avg_FHS': 48.2492,
                                        'avg_ES': 53.8037,
                                        'bt_newloan_bad': 0.0192,
                                        'bt_active': 0.3671,
                                        'fwd_positive_per_100_calls': 0.9513,
                                        'median_max_offer': 5000000.0,
                                        'fwd_collections': 0.1784},
                  'Credit-Hungry Stressed': {'segment': 'PROTECT',
                                             'customers': 43261,
                                             'share': 0.0432,
                                             'avg_FHS': 20.4543,
                                             'avg_ES': 47.5684,
                                             'bt_newloan_bad': 0.0442,
                                             'bt_active': 0.4168,
                                             'fwd_positive_per_100_calls': 0.866,
                                             'median_max_offer': 2000000.0,
                                             'fwd_collections': 0.1764},
                  'Silent Risk': {'segment': 'PROTECT',
                                  'customers': 109810,
                                  'share': 0.1098,
                                  'avg_FHS': 44.815,
                                  'avg_ES': 19.722,
                                  'bt_newloan_bad': 0.0154,
                                  'bt_active': 0.1561,
                                  'fwd_positive_per_100_calls': 0.2722,
                                  'median_max_offer': 2000000.0,
                                  'fwd_collections': 0.057},
                  'New-to-Credit Explorers': {'segment': 'BUILD',
                                              'customers': 23795,
                                              'share': 0.0238,
                                              'avg_FHS': 45.7179,
                                              'avg_ES': 32.3495,
                                              'bt_newloan_bad': 0.1189,
                                              'bt_active': 0.3195,
                                              'fwd_positive_per_100_calls': 1.6961,
                                              'median_max_offer': 21203000.0,
                                              'fwd_collections': 0.2151},
                  'Credit-Hungry Newcomers': {'segment': 'PROTECT',
                                              'customers': 32632,
                                              'share': 0.0326,
                                              'avg_FHS': 31.06,
                                              'avg_ES': 23.6904,
                                              'bt_newloan_bad': 0.0695,
                                              'bt_active': 0.1724,
                                              'fwd_positive_per_100_calls': 0.8532,
                                              'median_max_offer': 14000000.0,
                                              'fwd_collections': 0.1922},
                  'Dormant Unknowns': {'segment': 'BUILD',
                                       'customers': 125993,
                                       'share': 0.1259,
                                       'avg_FHS': 47.7415,
                                       'avg_ES': 15.424,
                                       'bt_newloan_bad': 0.0441,
                                       'bt_active': 0.1613,
                                       'fwd_positive_per_100_calls': 0.3817,
                                       'median_max_offer': 2000000.0,
                                       'fwd_collections': 0.1049}},
 'sub_evidence': {'Prime Champions': {'n': 24344, 'ci_low': 0.0107, 'ci_high': 0.0135},
                  'Loyal Core': {'n': 20244, 'ci_low': 0.0075, 'ci_high': 0.0101},
                  'Sleeping Prime': {'n': 162585, 'ci_low': 0.0064, 'ci_high': 0.0072},
                  'Cautious Growers': {'n': 7093, 'ci_low': 0.0164, 'ci_high': 0.0228},
                  'Stretched Seekers': {'n': 4266, 'ci_low': 0.0151, 'ci_high': 0.0233},
                  'Credit-Hungry Stressed': {'n': 3594, 'ci_low': 0.0375, 'ci_high': 0.051},
                  'Silent Risk': {'n': 19866, 'ci_low': 0.0137, 'ci_high': 0.0171},
                  'New-to-Credit Explorers': {'n': 757, 'ci_low': 0.0958, 'ci_high': 0.1419},
                  'Credit-Hungry Newcomers': {'n': 820, 'ci_low': 0.0521, 'ci_high': 0.0869},
                  'Dormant Unknowns': {'n': 13554, 'ci_low': 0.0407, 'ci_high': 0.0476}},
 'call_alloc': {'GROW': {'attempt_share': 0.7452, 'customer_share': 0.5106, 'pos_10k': 78.15},
                'REACTIVATE': {'attempt_share': 0.073, 'customer_share': 0.129, 'pos_10k': 29.19},
                'PROTECT': {'attempt_share': 0.1587, 'customer_share': 0.2107, 'pos_10k': 67.9},
                'BUILD': {'attempt_share': 0.0231, 'customer_share': 0.1497, 'pos_10k': 68.64}},
 'call_alloc_sub': {'Prime Champions': {'attempt_share': 0.3985, 'pos_10k': 104.28},
                    'Loyal Core': {'attempt_share': 0.2258, 'pos_10k': 39.8},
                    'Sleeping Prime': {'attempt_share': 0.073, 'pos_10k': 29.19},
                    'Cautious Growers': {'attempt_share': 0.1209, 'pos_10k': 63.71},
                    'Stretched Seekers': {'attempt_share': 0.0439, 'pos_10k': 95.13},
                    'Credit-Hungry Stressed': {'attempt_share': 0.0528, 'pos_10k': 86.6},
                    'Silent Risk': {'attempt_share': 0.0562, 'pos_10k': 27.22},
                    'New-to-Credit Explorers': {'attempt_share': 0.0053, 'pos_10k': 169.61},
                    'Credit-Hungry Newcomers': {'attempt_share': 0.0058, 'pos_10k': 85.32},
                    'Dormant Unknowns': {'attempt_share': 0.0177, 'pos_10k': 38.17}},
 'nba_mix': {'GROW · Pre-approved upsell / cross-sell': 0.2225,
             'PROTECT · No credit push – support & early warning': 0.1938,
             'T3 · Maturity-window re-loan': 0.1586,
             'BUILD · Starter product when they come': 0.1493,
             'GROW · Right-sized offer (structured)': 0.1052,
             'REACTIVATE · Digital win-back (no telesales)': 0.091,
             'T2 · Onboard & activate (60-day cool-off)': 0.0637,
             'T1 · Collections care & hardship plan': 0.016},
 'trigger_mix': {'None': 0.7617, 'T3 Loan maturing': 0.1586, 'T2 Just booked': 0.0637, 'T1 Live collections': 0.016},
 'channel_mix': {'SMS / Zalo OA': 0.4997,
                 'In-app / partner store (no outbound calls)': 0.1493,
                 'In-app / SMS (no telesales push)': 0.1052,
                 'Telesales call (best hours 8h, 9h, 10h; max 5 attempts, then SMS)': 0.0974,
                 'SMS / Zalo OA + in-app (no calls)': 0.0909,
                 'In-app (GMA) push + banner': 0.0414,
                 'Collections agent call + SMS': 0.016,
                 'In-app / SMS only (opted out of calls)': 0.0001},
 'value_levers': {'L1_calls_saved_cut30pct_lowES': 820500.6,
                  'L1_positives_kept': 0.8676,
                  'L2_healthy_customers_maturing_90d': 177252,
                  'L7_cautious_growers_structured': 112984,
                  'L7_bad_rate_cautious_growers': 0.0196,
                  'L7_bad_rate_stretched_seekers': 0.0192,
                  'L8_credit_hungry_newcomers': 32632,
                  'L8_bad_rate_credit_hungry_newcomers': 0.0695,
                  'L8_bad_rate_new_to_credit_explorers': 0.1189,
                  'L3_protect_customers': 210792,
                  'L3_bad_rate_protect': 0.0212,
                  'L3_bad_rate_grow': 0.0118,
                  'L3_protect_vs_grow_risk_ratio': 1.7897,
                  'L3_bad_loans_avoided_per_1000_paused_offers_protect': 21.1588,
                  'L3_credit_hungry_customers': 43261,
                  'L3_bad_rate_credit_hungry': 0.0442,
                  'L3_sales_calls_to_protect_share': 0.1587,
                  'L4_sales_calls_to_reactivate_build_share': 0.0961,
                  'L5_growth_actions': 726854,
                  'L6_protect_actions': 209895,
                  'forward_sales_attempts_total': 2735002.0},
 'matrix_share': {'Healthy (Resilient + Stable)': {'High': 0.2086, 'Medium': 0.1891, 'Low': 0.129},
                  'Vulnerable': {'High': 0.0526, 'Medium': 0.0854, 'Low': 0.0911},
                  'Distressed': {'High': 0.0299, 'Medium': 0.0133, 'Low': 0.0187},
                  'Thin-file': {'High': 0.0086, 'Medium': 0.0279, 'Low': 0.1459}},
 'gains': {'calls_cut_share': 0.3,
           'positives_kept_if_cut_30pct_lowest_ES': 0.8676352302433139,
           'positives_in_top20pct_calls': 0.382465312688518,
           'productivity_per_call_gain_if_cut': 0.23947890034759145,
           'priority_score_top20': 0.22486426704202694},
 'priority_validation': {'growth_calls_top20_by_priority_score': 0.27508820070560563, 'growth_calls_top20_by_ES': 0.4596236769894159},
 'audit_patch': {'F1_protect_ceiling_set_to_0': 124006,
                 'F2_credit_action_without_limit': 122292,
                 'F3_protect_T2_no_cross_sell': 13999,
                 'F4_timing_48h_replaced': 19589,
                 'F5_structured_without_down_payment': 47234,
                 'F6_silent_clean_explained': 86661,
                 'F7_moved_from_telesales_to_digital': 157836,
                 'F9_loyal_core_reason': 112096},
 'output_checks': {'rows': 1000528,
                   'one_row_per_customer': True,
                   'nulls_in_required_columns': 0,
                   'empty_text_in_required_columns': 0,
                   'explanation_names_its_action_pct': 1.0,
                   'T3_outside_Resilient_Stable': 0,
                   'credit_offer_to_PROTECT': 0,
                   'credit_ceiling_on_PROTECT': 0,
                   'stressed_customers_in_GROW_or_REACTIVATE': 0,
                   'telesales_to_REACTIVATE_BUILD_PROTECT': 0},
 'best_call_hours': [8, 9, 10],
 'run_id': '2026-10-18 05:16:34',
 'nba_by_segment': {'BUILD': {'BUILD · Starter product when they come': 0.997,
                              'GROW · Pre-approved upsell / cross-sell': 0.0,
                              'GROW · Right-sized offer (structured)': 0.0,
                              'PROTECT · No credit push – support & early warning': 0.0,
                              'REACTIVATE · Digital win-back (no telesales)': 0.0,
                              'T1 · Collections care & hardship plan': 0.003,
                              'T2 · Onboard & activate (60-day cool-off)': 0.0,
                              'T3 · Maturity-window re-loan': 0.0},
                    'GROW': {'BUILD · Starter product when they come': 0.0,
                             'GROW · Pre-approved upsell / cross-sell': 0.436,
                             'GROW · Right-sized offer (structured)': 0.206,
                             'PROTECT · No credit push – support & early warning': 0.0,
                             'REACTIVATE · Digital win-back (no telesales)': 0.0,
                             'T1 · Collections care & hardship plan': 0.023,
                             'T2 · Onboard & activate (60-day cool-off)': 0.097,
                             'T3 · Maturity-window re-loan': 0.237},
                    'PROTECT': {'BUILD · Starter product when they come': 0.0,
                                'GROW · Pre-approved upsell / cross-sell': 0.0,
                                'GROW · Right-sized offer (structured)': 0.0,
                                'PROTECT · No credit push – support & early warning': 0.92,
                                'REACTIVATE · Digital win-back (no telesales)': 0.0,
                                'T1 · Collections care & hardship plan': 0.014,
                                'T2 · Onboard & activate (60-day cool-off)': 0.066,
                                'T3 · Maturity-window re-loan': 0.0},
                    'REACTIVATE': {'BUILD · Starter product when they come': 0.0,
                                   'GROW · Pre-approved upsell / cross-sell': 0.0,
                                   'GROW · Right-sized offer (structured)': 0.0,
                                   'PROTECT · No credit push – support & early warning': 0.0,
                                   'REACTIVATE · Digital win-back (no telesales)': 0.705,
                                   'T1 · Collections care & hardship plan': 0.006,
                                   'T2 · Onboard & activate (60-day cool-off)': 0.0,
                                   'T3 · Maturity-window re-loan': 0.29}},
 'explanations': {'Prime Champions': 'GROW | FHS 95 (Resilient; + Loans successfully taken (track record): 5+; − No material negative '
                                     'factor) | ES 41 (Active; last self-initiated activity 143d ago) → T3 · Maturity-window re-loan: '
                                     'healthy and a loan is about to mature – the moment repeat demand peaks',
                  'Loyal Core': 'GROW | FHS 67 (Stable; + Past bad loans: 0; − Applications last 90d: 1) | ES 35 (Warm; last '
                                'self-initiated activity 60d ago) → T3 · Maturity-window re-loan: healthy and a loan is about to mature – '
                                'the moment repeat demand peaks',
                  'Sleeping Prime': 'REACTIVATE | FHS 74 (Resilient; + Past bad loans: 0; − No material negative factor) | ES 23 (Cooling; '
                                    'last self-initiated activity 162d ago) → REACTIVATE · Digital win-back (no telesales): healthy but '
                                    'silent – good risk, rarely answers calls; win back through a cheap digital channel',
                  'Cautious Growers': 'GROW | FHS 54 (Vulnerable; + Past bad loans: 0; − Loans successfully taken (track record): 1) | ES '
                                      '40 (Active; last self-initiated activity 128d ago) → GROW · Right-sized offer (structured): clean '
                                      'but short history and engaged – grow with a smaller, shorter, safer loan (C1: grow with structure)',
                  'Stretched Seekers': 'PROTECT | FHS 49 (Vulnerable; + Loans successfully taken (track record): 5+; − Rejections last '
                                       '12m: 2+) | ES 65 (Champion; last self-initiated activity 41d ago) → T2 · Onboard & activate '
                                       '(60-day cool-off): just took a loan – a new offer now adds risk and rarely converts; activation '
                                       'builds engagement',
                  'Credit-Hungry Stressed': 'PROTECT | FHS 38 (Distressed; + Past bad loans: 0; − Rejections last 12m: 2+) | ES 29 (Warm; '
                                            'last self-initiated activity 13d ago) → PROTECT · No credit push – support & early warning: '
                                            'fragile and actively hunting credit – more credit would deepen stress',
                  'Silent Risk': 'PROTECT | FHS 0 (Distressed; + Relationship tenure: 5y+; − Past bad loans: 1) | ES 16 (Cooling; last '
                                 'self-initiated activity 130d ago) → PROTECT · No credit push – support & early warning: fragile and '
                                 'silent – prevent arrears, do not sell',
                  'New-to-Credit Explorers': 'BUILD | FHS 40 (Thin-file; + No negative record; − No credit history with us (thin file)) | '
                                             'ES 35 (Warm; last self-initiated activity 47d ago) → BUILD · Starter product when they come: '
                                             'no credit history yet – a small first product builds the repayment record we need',
                  'Credit-Hungry Newcomers': 'PROTECT | FHS 34 (Thin-file; + No negative record; − No credit history with us (thin file)) '
                                             '| ES 21 (Cooling; last self-initiated activity 114d ago) → PROTECT · No credit push – '
                                             'support & early warning: no history yet and already hunting credit hard – the riskiest first '
                                             'loans; verify before lending',
                  'Dormant Unknowns': 'BUILD | FHS 46 (Thin-file; + No negative record; − No credit history with us (thin file)) | ES 13 '
                                      '(Cooling; last self-initiated activity 175d ago) → BUILD · Starter product when they come: no '
                                      'credit history yet – a small first product builds the repayment record we need'}}


# =============================================================================================================
# Design system — Home Credit / consulting palette. All ink is navy-to-charcoal so it stays AAA-readable on
# the white report cards no matter what colour scheme the browser is running.
# =============================================================================================================
RED, NAVY, GREEN, BLUE, AMBER, PURPLE, SLATE = '#E11931', '#0F2043', '#107C41', '#2563EB', '#D97706', '#7C3AED', '#64748B'
NAVY2, CANVAS, LIGHT, GRID = '#1B3A6B', '#E9EEF6', '#F4F6F9', '#E5E7EB'

# --- the four ink levels used everywhere (no light greys on white any more) ---
INK = '#0F2043'        # headline / primary ink (navy)
INK_TICK = '#1E293B'   # axis tick labels
INK_BODY = '#0F172A'   # charcoal — subtitles, captions, take-aways, table cells
INK_SOFT = '#334155'   # sharp dark grey — footnotes, secondary labels
GRID_C = '#E2E8F0'     # chart grid
ZERO_C = '#CBD5E1'     # zero line / control borders

TIER_ORDER = ['Resilient', 'Stable', 'Vulnerable', 'Distressed', 'Thin-file']
TIER_COLOR = {'Resilient': GREEN, 'Stable': BLUE, 'Vulnerable': AMBER, 'Distressed': RED, 'Thin-file': SLATE}
SEG_ORDER = ['GROW', 'REACTIVATE', 'PROTECT', 'BUILD']
SEG_COLOR = {'GROW': GREEN, 'REACTIVATE': PURPLE, 'PROTECT': RED, 'BUILD': AMBER}
LEVEL_ORDER = ['Cooling', 'Warm', 'Active', 'Champion']
LEVEL_COLOR = {'Cooling': '#7C8DA4', 'Warm': '#60A5FA', 'Active': BLUE, 'Champion': NAVY}
SUB_ORDER = ['Prime Champions', 'Loyal Core', 'Cautious Growers', 'Sleeping Prime', 'Stretched Seekers', 'Credit-Hungry Stressed',
             'Silent Risk', 'Credit-Hungry Newcomers', 'New-to-Credit Explorers', 'Dormant Unknowns']
TRIGGER_ORDER = ['T1 Live collections', 'T2 Just booked', 'T3 Loan maturing', 'None']
N_TOTAL = 1_000_528
SRC = 'Source: BI10 dataset · Customer Intelligence Engine final run (' + OFFICIAL['run_id'] + ') · summary_metrics.json'
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Short, board-ready labels for the 8 Next-Best-Actions. The full sentence goes into the hover tooltip only,
# so a long y-axis label never has to be cut off with "..." again.
ACTION_SHORT = {
    'GROW · Pre-approved upsell / cross-sell': 'GROW · Pre-approved offer',
    'PROTECT · No credit push – support & early warning': 'PROTECT · No credit push',
    'T3 · Maturity-window re-loan': 'T3 · Maturity re-loan',
    'BUILD · Starter product when they come': 'BUILD · Starter product',
    'GROW · Right-sized offer (structured)': 'GROW · Structured offer',
    'REACTIVATE · Digital win-back (no telesales)': 'REACTIVATE · Digital win-back',
    'T2 · Onboard & activate (60-day cool-off)': 'T2 · Onboard (60d cool-off)',
    'T1 · Collections care & hardship plan': 'T1 · Collections care'}


def short_action(a):
    """Compact y-axis label; an unknown action degrades gracefully instead of raising."""
    a = str(a)
    return ACTION_SHORT.get(a, a.split(' – ')[0].split(' (')[0][:34])


# --- Streamlit / Plotly capability probes: the app must run on old and new versions alike ---------------------
try:
    _V = tuple(int(x) for x in st.__version__.split('.')[:2])
except Exception:
    _V = (1, 0)
FULL = {'width': 'stretch'} if _V >= (1, 46) else {'use_container_width': True}
CAN_SELECT = _V >= (1, 35)           # st.plotly_chart(on_select=...)
try:                                 # layout.font.weight only exists on newer Plotly
    go.Figure().update_layout(xaxis=dict(title_font=dict(weight='bold')))
    _BOLD = {'weight': 'bold'}
except Exception:
    _BOLD = {}

st.set_page_config(page_title='Customer Intelligence Engine · FT-US', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

# =============================================================================================================
# CSS hard-lock. Streamlit inherits the OS / browser colour scheme, which turns select boxes black and
# secondary text light grey. Everything below pins the report canvas to a white surface with navy ink.
# Note: BaseWeb renders drop-down menus in a portal attached to <body>, so those rules are deliberately global.
# =============================================================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ---------- 0 - refuse the browser's dark mode outright ---------- */
  :root, html, body, .stApp, section[data-testid="stMain"] { color-scheme: light only !important; }
  html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
  .stApp { background: #E9EEF6 !important; }
  header[data-testid="stHeader"] { background: transparent; height: 0; }
  div[data-testid="stToolbar"], div[data-testid="stDecoration"] { display: none; }
  .block-container { padding: 1.0rem 1.6rem 1.2rem 1.6rem; max-width: 1680px; }

  /* ---------- 1 - default ink for the whole report canvas (AAA contrast on white) ---------- */
  .stApp, .stApp p, .stApp li, .stApp label, .stApp small, .stApp td, .stApp th,
  div[data-testid="stMarkdownContainer"], div[data-testid="stMarkdownContainer"] p,
  div[data-testid="stExpander"] summary,
  div[data-testid="stAlert"], div[data-testid="stAlert"] * { color: #0F172A; }
  div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] * { color: #334155 !important; font-weight: 500; }

  /* ---------- 2 - SELECT BOXES: white surface, navy ink, blue focus ring ---------- */
  div[data-baseweb="select"] > div,
  div[data-baseweb="select"] div[data-baseweb="input"],
  div[data-baseweb="base-input"] {
      background-color: #FFFFFF !important; background-image: none !important;
      border: 1px solid #CBD5E1 !important; border-radius: 9px !important;
      color: #0F2043 !important; font-weight: 600 !important; box-shadow: none !important; }
  div[data-baseweb="select"] > div > div, div[data-baseweb="select"] span, div[data-baseweb="select"] div,
  div[data-baseweb="select"] input {
      color: #0F2043 !important; -webkit-text-fill-color: #0F2043 !important; font-weight: 600 !important; }
  div[data-baseweb="select"] svg { fill: #334155 !important; color: #334155 !important; }
  div[data-baseweb="select"] > div:hover { border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important; }
  div[data-baseweb="select"] > div:focus-within,
  div[data-baseweb="select"] > div[data-focusvisible="true"] { border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.20) !important; }

  /* ---------- 3 - DROP-DOWN MENUS (rendered in a body-level portal) ---------- */
  div[data-baseweb="popover"], div[data-baseweb="popover"] > div,
  div[data-baseweb="popover"] div[data-baseweb="menu"], ul[role="listbox"] {
      background-color: #FFFFFF !important; background-image: none !important;
      border-radius: 10px !important; box-shadow: 0 12px 30px rgba(15,32,67,.18) !important; }
  div[data-baseweb="popover"] div[data-baseweb="menu"], ul[role="listbox"] { border: 1px solid #CBD5E1 !important; padding: 4px !important; }
  ul[role="listbox"] li, li[role="option"], div[data-baseweb="popover"] li {
      background-color: #FFFFFF !important; color: #0F2043 !important; font-weight: 600 !important; border-radius: 7px !important; }
  li[role="option"]:hover, li[role="option"][aria-selected="true"],
  ul[role="listbox"] li:hover { background-color: #EFF6FF !important; color: #0F2043 !important; }
  div[data-baseweb="popover"] *, ul[role="listbox"] * { color: #0F2043 !important; -webkit-text-fill-color: #0F2043 !important; }

  /* ---------- 4 - TEXT / NUMBER INPUTS & MULTISELECT ---------- */
  div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
  .stTextInput input, .stNumberInput input, .stTextArea textarea {
      background-color: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 9px !important;
      color: #0F2043 !important; -webkit-text-fill-color: #0F2043 !important; font-weight: 600 !important; }
  div[data-testid="stTextInput"] input:focus, div[data-testid="stNumberInput"] input:focus, .stTextArea textarea:focus {
      border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.20) !important; outline: none !important; }
  input::placeholder, textarea::placeholder { color: #64748B !important; opacity: 1 !important; font-weight: 500 !important; }
  div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div { background-color: #FFFFFF !important; }
  span[data-baseweb="tag"] { background-color: #0F2043 !important; border-radius: 7px !important; }
  span[data-baseweb="tag"], span[data-baseweb="tag"] span, span[data-baseweb="tag"] div {
      color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 600 !important; }
  span[data-baseweb="tag"] svg { fill: #FFFFFF !important; }

  /* ---------- 4b - THE SAME CONTROLS ON NEWER STREAMLIT (React Aria replaced BaseWeb) ----------
     Streamlit >= ~1.50 renders select boxes as .react-aria-ComboBox with an inner div[role="group"]
     and no longer emits data-baseweb at all. Both DOM generations are pinned here, so the dashboard
     looks identical on an older 1.3x deployment and on the newest Streamlit.                          */
  .react-aria-ComboBox > div[role="group"],
  div[data-testid="stSelectbox"] .react-aria-ComboBox > div,
  div[data-testid="stMultiSelect"] .react-aria-ComboBox > div {
      background-color: #FFFFFF !important; background-image: none !important;
      border: 1px solid #CBD5E1 !important; border-radius: 9px !important; }
  .react-aria-ComboBox > div[role="group"]:hover {
      border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important; }
  .react-aria-ComboBox > div[role="group"]:focus-within {
      border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,.20) !important; }
  .react-aria-ComboBox input, .react-aria-ComboBox span, .react-aria-ComboBox div[role="group"] > div {
      color: #0F2043 !important; -webkit-text-fill-color: #0F2043 !important; font-weight: 600 !important; }
  .react-aria-ComboBox button svg, .react-aria-ComboBox svg { fill: #334155 !important; color: #334155 !important; }
  /* the drop-down list itself, whichever element carries the ARIA role */
  [role="listbox"] {
      background-color: #FFFFFF !important; background-image: none !important;
      border: 1px solid #CBD5E1 !important; border-radius: 10px !important;
      box-shadow: 0 12px 30px rgba(15,32,67,.18) !important; padding: 4px !important; }
  [role="option"] {
      background-color: #FFFFFF !important; color: #0F2043 !important;
      -webkit-text-fill-color: #0F2043 !important; font-weight: 600 !important; border-radius: 7px !important; }
  [role="option"] * { color: #0F2043 !important; -webkit-text-fill-color: #0F2043 !important; }
  [role="option"]:hover, [role="option"][aria-selected="true"],
  [role="option"][data-hovered], [role="option"][data-focused], [role="option"][data-selected] {
      background-color: #EFF6FF !important; color: #0F2043 !important; }
  /* help tooltips also live in the portal: keep them light instead of Streamlit's dark bubble */
  div[data-testid="stTooltipContent"], div[data-testid="stTooltipContent"] * {
      background-color: #FFFFFF !important; color: #0F172A !important; }

  /* ---------- 5 - SLIDERS, RADIOS, BUTTONS, TABLES on the main canvas ---------- */
  div[data-testid="stSlider"] label, div[data-testid="stSlider"] label p { color: #0F2043 !important; font-weight: 600 !important; }
  div[data-testid="stTickBar"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"],
  div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { color: #334155 !important; font-weight: 600 !important; }
  div[data-baseweb="slider"] div[role="slider"] { box-shadow: 0 1px 4px rgba(15,32,67,.25) !important; }
  div[data-testid="stSlider"] div[data-testid="stThumbValue"] { color: #0F2043 !important; font-weight: 700 !important; }
  section[data-testid="stMain"] div[role="radiogroup"] label p { color: #0F2043 !important; font-weight: 600 !important; }
  section[data-testid="stMain"] button, div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button {
      background-color: #FFFFFF !important; color: #0F2043 !important; border: 1px solid #CBD5E1 !important;
      border-radius: 9px !important; font-weight: 600 !important; }
  section[data-testid="stMain"] button:hover, div[data-testid="stDownloadButton"] button:hover, div[data-testid="stButton"] button:hover {
      border-color: #2563EB !important; color: #1D4ED8 !important; box-shadow: 0 2px 8px rgba(37,99,235,.16) !important; }
  section[data-testid="stMain"] button p, div[data-testid="stButton"] button p, div[data-testid="stDownloadButton"] button p {
      color: inherit !important; font-weight: 600 !important; }
  div[data-testid="stDataFrame"] { background: #FFFFFF !important; border-radius: 10px; }
  div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * { color: #0F2043 !important; font-weight: 800 !important; }
  div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color: #334155 !important; font-weight: 600 !important; }
  div[data-testid="stMetric"] { background: #F4F6F9; border-radius: 10px; padding: 8px 12px; border: 1px solid #E8EDF5; }
  .stSelectbox label, .stSelectbox label p, .stMultiSelect label, .stMultiSelect label p,
  .stSlider label, .stRadio label, .stTextInput label, .stTextInput label p {
      color: #0F2043 !important; font-weight: 600 !important; font-size: .78rem !important; }

  /* ---------- 6 - white report cards ---------- */
  div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] span.cardmark),
  div[data-testid="stVerticalBlock"]:has(> div.element-container span.cardmark) {
      background: #FFFFFF; border-radius: 14px; padding: 12px 14px 10px 14px;
      box-shadow: 0 1px 3px rgba(15,32,67,.10), 0 1px 2px rgba(15,32,67,.05); height: 100%; }
  div[data-testid="stElementContainer"]:has(span.cardmark), div.element-container:has(span.cardmark) { display: none; }

  /* ---------- 7 - typography of the report itself ---------- */
  .pg-num { color: #2563EB; font-weight: 800; font-size: 2.1rem; margin-right: 10px; }
  .pg-title { color: #0F2043; font-weight: 800; font-size: 2.1rem; letter-spacing: -.01em; }
  .pg-sub { color: #334155; font-size: .92rem; margin-top: -4px; font-weight: 500; }
  .kpi { background: #FFFFFF; border-radius: 12px; padding: 0 0 12px 0; box-shadow: 0 1px 3px rgba(15,32,67,.08); overflow: hidden; height: 100%; }
  .kpi .bar { height: 5px; background: #0F2043; }
  .kpi .lab { color: #0F2043; font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; padding: 10px 14px 0 14px; }
  .kpi .num { font-size: 1.85rem; font-weight: 800; line-height: 1.15; padding: 2px 14px 0 14px; color: #0F2043; }
  .kpi .sub { color: #0F172A; font-size: .78rem; line-height: 1.35; padding: 3px 14px 0 14px; font-weight: 500; }
  .kpi .pillrow { padding: 6px 14px 0 14px; }
  .pill { display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 700; font-size: .70rem; white-space: nowrap; }
  .pill-g { background: #E6F4EC; color: #0B6B38; border: 1px solid #B7E0C8; }
  .pill-r { background: #FDE8EB; color: #A5101F; border: 1px solid #F6C2CA; }
  .pill-b { background: #E8F0FE; color: #1D4ED8; border: 1px solid #C3D7FB; }
  .pill-a { background: #FDF1DD; color: #8A5204; border: 1px solid #F5DCB3; }
  .pill-s { background: #EEF2F7; color: #334155; border: 1px solid #D8E0EA; }
  .ct { color: #0F2043; font-weight: 700; font-size: .98rem; text-align: center; margin: 0; }
  .cs { color: #334155; font-size: .78rem; text-align: center; margin: 0 0 2px 0; font-weight: 500; }
  .src { color: #334155; font-size: .71rem; font-style: italic; margin: -2px 0 0 0; }
  .take { background: #FFFFFF; border-radius: 12px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(15,32,67,.08); height: 100%; }
  .take h4 { color: #0F2043; margin: 0 0 6px 0; font-size: 1rem; }
  .take .it { margin: 6px 0; font-size: .84rem; color: #0F172A; line-height: 1.4; }
  .take .it b { color: #C01026; }
  .tag { display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 700; font-size: .74rem; margin-right: 5px; }
  .tag, .tag * { color: #FFFFFF !important; }
  table.tbl { width: 100%; border-collapse: collapse; font-size: .80rem; color: #0F172A; }
  table.tbl th { background: #0F2043; text-align: left; padding: 6px 8px; font-weight: 600; }
  table.tbl th, table.tbl th * { color: #FFFFFF !important; }
  table.tbl td { padding: 5px 8px; border-bottom: 1px solid #E5E7EB; vertical-align: top; color: #0F172A; }
  table.tbl tr:nth-child(even) td { background: #F4F6F9; }
  .grid td { padding: 7px 9px; border: 3px solid #FFFFFF; font-weight: 600; font-size: .80rem; vertical-align: top; border-radius: 6px; }
  .grid td, .grid td * { color: #FFFFFF !important; }
  .grid th { padding: 4px 8px; color: #0F2043 !important; font-size: .76rem; text-align: left; }
  .grid td.rowh { background: #FFFFFF; font-weight: 700; }
  .grid td.rowh, .grid td.rowh * { color: #0F2043 !important; }
  .big { border-radius: 14px; background: #FFFFFF; padding: 0; overflow: hidden; box-shadow: 0 1px 3px rgba(15,32,67,.08); height: 100%; }
  .big .hd { background: #0F2043; padding: 8px 14px; font-weight: 700; font-size: .85rem; }
  .big .hd, .big .hd * { color: #FFFFFF !important; }
  .big .n { font-size: 2.6rem; font-weight: 800; color: #C01026; padding: 6px 14px 0 14px; }
  .big .t { padding: 0 14px 12px 14px; color: #0F172A; font-size: .84rem; line-height: 1.4; }
  .hero { border-radius: 14px; padding: 14px 20px; margin: 10px 0 12px 0; background: #0F2043; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .hero, .hero * { color: #FFFFFF !important; }
  .expl { background: #F4F6F9; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px;
          font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .78rem; color: #0F172A; white-space: pre-wrap; line-height: 1.5; }
  .fbanner { background: linear-gradient(90deg,#EFF6FF,#F8FAFF); border: 1px solid #BFD7FE; border-left: 5px solid #2563EB;
             border-radius: 10px; padding: 9px 14px; color: #0F2043; font-size: .84rem; font-weight: 600; }
  .fbanner b { color: #1D4ED8; }
  .rule { background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 4px solid #107C41; border-radius: 9px;
          padding: 8px 12px; margin: 5px 0; color: #0F172A; font-size: .80rem; line-height: 1.45; }
  .rule b { color: #0F2043; }

  /* ---------- 8 - navy navigation rail (declared last so its light ink always wins) ---------- */
  section[data-testid="stSidebar"] { background: #0F2043 !important; min-width: 242px !important; max-width: 242px !important; }
  section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] [role="radiogroup"] label { padding: 9px 12px; border-radius: 8px; margin: 2px 0; width: 100%; }
  section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #1B3A6B; }
  section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: #2B4C85; box-shadow: inset 3px 0 0 #E11931; }
  section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child,
  section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child:not(:has(p)) { display: none; }
  section[data-testid="stSidebar"] .stRadio p { font-size: .92rem; font-weight: 600; }
  section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #1B3A6B !important; border: 1px dashed #47628F !important; }
  section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #CBD5E1 !important; }
  section[data-testid="stSidebar"] button { background: #1B3A6B !important; border: 1px solid #47628F !important; }
  section[data-testid="stSidebar"] button * { color: #E2E8F0 !important; }
  section[data-testid="stSidebar"] hr { border-color: #2B4C85; }
</style>""", unsafe_allow_html=True)


# =============================================================================================================
# THE theme function — every Plotly figure in this app goes through it, and show() always renders with
# theme=None, so Streamlit's dark template can never touch tick fonts, axis titles or legends.
# =============================================================================================================
def apply_pro_theme(fig, height=280, legend=True, bargap=0.28, margin=None, xtitle=None, ytitle=None):
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=margin or dict(l=6, r=12, t=10, b=6),
        showlegend=legend,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, Segoe UI, sans-serif', color=INK, size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='left', x=0, title=None,
                    font=dict(family='Inter, Segoe UI, sans-serif', color=INK, size=11),
                    bgcolor='rgba(255,255,255,0.85)', bordercolor=GRID_C, borderwidth=0),
        hoverlabel=dict(bgcolor=INK, font_size=12, font_color='#FFFFFF', bordercolor=INK,
                        font_family='Inter, Segoe UI, sans-serif', align='left'),
        bargap=bargap)
    fig.update_xaxes(
        tickfont=dict(color=INK_TICK, size=11, family='Inter, Segoe UI, sans-serif'),
        title_font=dict(color=INK, size=12, family='Inter, Segoe UI, sans-serif', **_BOLD),
        gridcolor=GRID_C, zerolinecolor=ZERO_C, linecolor=ZERO_C, automargin=True)
    fig.update_yaxes(
        tickfont=dict(color=INK_TICK, size=11, family='Inter, Segoe UI, sans-serif'),
        title_font=dict(color=INK, size=12, family='Inter, Segoe UI, sans-serif', **_BOLD),
        gridcolor=GRID_C, zerolinecolor=ZERO_C, linecolor=ZERO_C, automargin=True)
    if xtitle:
        fig.update_xaxes(title=dict(text=xtitle))
    if ytitle:
        fig.update_yaxes(title=dict(text=ytitle))
    # subplot titles / free annotations are separate objects: force them to navy as well
    fig.update_annotations(font=dict(color=INK, family='Inter, Segoe UI, sans-serif'))
    try:                                      # colour-bar ticks, when a figure has one
        fig.update_coloraxes(colorbar=dict(tickfont=dict(color=INK_TICK, size=10)))
    except Exception:
        pass
    return fig


# alias, so that every historical call site is themed too
style = apply_pro_theme


def apply_gauge_theme(fig, height=210):
    """go.Indicator figures have no cartesian axes — they need their own (equally hard-locked) pass."""
    fig.update_layout(height=height, margin=dict(l=18, r=18, t=40, b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter, Segoe UI, sans-serif', color=INK, size=12),
                      hoverlabel=dict(bgcolor=INK, font_size=12, font_color='#FFFFFF', bordercolor=INK))
    return fig


def show(fig, key, select=False, config=None):
    """The ONLY way a chart reaches the page. theme=None is mandatory: it forbids Streamlit's dark override."""
    cfg = {'displayModeBar': False, 'displaylogo': False}
    if config:
        cfg.update(config)
    if select and CAN_SELECT:
        return st.plotly_chart(fig, **FULL, config=cfg, key=key, theme=None, on_select='rerun', selection_mode='points')
    st.plotly_chart(fig, **FULL, config=cfg, key=key, theme=None)
    return None


def chart_points(event):
    """Read a plotly_chart selection without ever raising — works on every Streamlit version and every shape
    (attribute object, plain dict, None, missing 'points', empty list)."""
    try:
        if event is None:
            return []
        sel = event.get('selection') if isinstance(event, dict) else getattr(event, 'selection', None)
        if sel is None:
            return []
        pts = sel.get('points', []) if isinstance(sel, dict) else getattr(sel, 'points', [])
        return list(pts or [])
    except Exception:
        return []


def point_field(point, field):
    """Safe field read on one selected point (dict-like or attribute-like)."""
    try:
        if isinstance(point, dict):
            return point.get(field)
        return getattr(point, field, None)
    except Exception:
        return None


# =============================================================================================================
# Small formatting / layout helpers
# =============================================================================================================
def pct(x, d=1):
    return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x * 100:.{d}f}%'


def vnd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    x = float(x)
    return f'{x / 1e6:,.1f}M VND' if x >= 1e6 else f'{x:,.0f} VND'


def kpi(label, number, sub, color=NAVY, bar=NAVY, pill=None, pill_kind='s'):
    """KPI card with an optional comparison badge (delta pill)."""
    badge = f'<div class="pillrow"><span class="pill pill-{pill_kind}">{pill}</span></div>' if pill else ''
    st.markdown(f'<div class="kpi"><div class="bar" style="background:{bar}"></div><div class="lab">{label}</div>'
                f'<div class="num" style="color:{color}">{number}</div>{badge}<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def stat(col, label, value, color=NAVY):
    col.markdown(f'<div style="background:{LIGHT};border:1px solid #E8EDF5;border-radius:10px;padding:7px 10px;">'
                 f'<div style="font-size:.72rem;color:{INK_SOFT};font-weight:600">{label}</div>'
                 f'<div style="font-size:1.3rem;font-weight:800;color:{color};line-height:1.2">{value}</div></div>', unsafe_allow_html=True)


def title(t, sub=None):
    st.markdown(f'<p class="ct">{t}</p>' + (f'<p class="cs">{sub}</p>' if sub else ''), unsafe_allow_html=True)


def source(text):
    st.markdown(f'<p class="src">{text}</p>', unsafe_allow_html=True)


def card():
    c = st.container()
    c.markdown('<span class="cardmark"></span>', unsafe_allow_html=True)
    return c


def table_html(df):
    st.markdown(df.to_html(index=False, escape=True, classes='tbl', border=0), unsafe_allow_html=True)


def takeaway(items, head='💡 Take away'):
    body = ''.join(f'<div class="it"><b>{h}</b><br>{t}</div>' for h, t in items)
    st.markdown(f'<div class="take"><h4>{head}</h4>{body}</div>', unsafe_allow_html=True)


def gap(h=10):
    st.markdown(f'<div style="height:{h}px"></div>', unsafe_allow_html=True)


def wshare(df, col, order):
    s = df.groupby(df[col].astype(str)).WEIGHT.sum()
    s = s.reindex(order).fillna(0)
    return s / max(s.sum(), 1e-9)


def wmean(df, col):
    v = df[col].astype(float)
    m = v.notna()
    return float(np.average(v[m], weights=df.WEIGHT[m])) if m.any() else float('nan')


# --- maths: monotone gains curve (no SciPy needed) ---------------------------------------------------------
def monotone_curve(xp, yp, x):
    """Fritsch–Carlson monotone cubic interpolation (no SciPy needed)."""
    xp, yp, x = np.asarray(xp, float), np.asarray(yp, float), np.atleast_1d(np.asarray(x, float))
    h = np.diff(xp); dlt = np.diff(yp) / h
    m = np.r_[dlt[0], (dlt[:-1] + dlt[1:]) / 2, dlt[-1]]
    for k in range(len(dlt)):
        if dlt[k] == 0:
            m[k] = m[k + 1] = 0
        else:
            a, b = m[k] / dlt[k], m[k + 1] / dlt[k]
            s = a * a + b * b
            if s > 9:
                t = 3 / np.sqrt(s); m[k], m[k + 1] = t * a * dlt[k], t * b * dlt[k]
    idx = np.clip(np.searchsorted(xp, x) - 1, 0, len(h) - 1)
    t = (x - xp[idx]) / h[idx]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return h00 * yp[idx] + h10 * h[idx] * m[idx] + h01 * yp[idx + 1] + h11 * h[idx] * m[idx + 1]


GAINS_X = [0.0, 0.20, 0.70, 1.0]
GAINS_Y = [0.0, OFFICIAL['gains']['positives_in_top20pct_calls'], OFFICIAL['gains']['positives_kept_if_cut_30pct_lowest_ES'], 1.0]


def gains(share_of_calls):
    return float(np.clip(monotone_curve(GAINS_X, GAINS_Y, share_of_calls)[0], 0, 1))


# =============================================================================================================
# Data engine: real file -> stratified sample (re-weighted) -> synthetic demo. Never raises, always opens.
# =============================================================================================================
REQUIRED = {
    'SKP_CLIENT': 0, 'FHS': np.nan, 'FHS_TIER': 'Thin-file', 'FHS_SUBGROUP': '', 'FHS_CONFIDENCE': 'Low', 'STRESS_FLAG': 0, 'LIVE_STRESS_FLAG': 0,
    'FHS_POS_REASON': '', 'FHS_NEG_REASON': '', 'FHS_NEG_GROUP': '', 'ES': np.nan, 'ES_LEVEL': 'Cooling', 'ES_RECENCY': np.nan,
    'ES_FREQUENCY': np.nan, 'ES_INTERACTION': np.nan, 'ES_LEAD_ACTIVITY': np.nan, 'ES_PRODUCT_INTERACTION': np.nan,
    'DAYS_SINCE_LAST_SELF_ACTIVITY': np.nan, 'SEGMENT': 'BUILD', 'SUB_SEGMENT': 'Dormant Unknowns', 'NBA_TRIGGER': 'None', 'NBA_ACTION': '',
    'NBA_DETAIL': '', 'NBA_PRODUCT': '', 'NBA_OFFER_VND': 0.0, 'CREDIT_CEILING_VND': 0.0, 'NBA_CHANNEL': '', 'NBA_TIMING': '',
    'NBA_PRIORITY': 'Low', 'NBA_PRIORITY_SCORE': 0.0, 'EXPLANATION': '', 'MAX_OFFER': np.nan, 'INCOME': np.nan, 'DTI_EXISTING': np.nan, 'PROVINCE': ''}
ALIASES = {'DTI': 'DTI_EXISTING', 'AMT_INCOME_MAIN': 'INCOME', 'MAX_OFFER_VND': 'MAX_OFFER'}

ACTION_LABEL = {
    'GROW_OFFER': 'GROW · Pre-approved upsell / cross-sell', 'GROW_STRUCTURED': 'GROW · Right-sized offer (structured)',
    'REACTIVATE': 'REACTIVATE · Digital win-back (no telesales)', 'PROTECT': 'PROTECT · No credit push – support & early warning',
    'BUILD': 'BUILD · Starter product when they come', 'T1': 'T1 · Collections care & hardship plan',
    'T2': 'T2 · Onboard & activate (60-day cool-off)', 'T3': 'T3 · Maturity-window re-loan'}
WHY = {
    'GROW_OFFER': 'healthy and engaged – lowest risk and highest response; deepen the relationship',
    'GROW_STRUCTURED': 'clean but short history and engaged – grow with a smaller, shorter, safer loan (C1: grow with structure)',
    'REACTIVATE': 'healthy but silent – good risk, rarely answers calls; win back through a cheap digital channel',
    'BUILD': 'no credit history yet – a small first product builds the repayment record we need',
    'T1': 'already in collections – protecting the customer and the book comes first',
    'T2': 'just took a loan – a new offer now adds risk and rarely converts; activation builds engagement',
    'T3': 'healthy and a loan is about to mature – the moment repeat demand peaks',
    'Credit-Hungry Stressed': 'fragile and actively hunting credit – more credit would deepen stress',
    'Stretched Seekers': 'engaged but showing stress signals (rejections, credit hunger) – no more credit for now',
    'Credit-Hungry Newcomers': 'no history yet and already hunting credit hard – the riskiest first loans; verify before lending',
    'Silent Risk': 'fragile and silent – prevent arrears, do not sell'}
TEL = 'Telesales call (best hours 8h, 9h, 10h; max 5 attempts, then SMS)'


def _tier_from_fhs(f):
    return 'Resilient' if f >= 70 else 'Stable' if f >= 55 else 'Vulnerable' if f >= 40 else 'Distressed'


def _level_from_es(e):
    c = OFFICIAL['es_cuts']
    return 'Cooling' if e < c[0] else 'Warm' if e < c[1] else 'Active' if e < c[2] else 'Champion'


def _persona_rows():
    """The exact customers quoted in the report / slides (explanation_examples of the final run)."""
    rows, rx = [], re.compile(r'^(\w+) \| FHS (\d+) \((.+?); \+ (.+?); − (.+?)\) \| ES (\d+) \((\w+); last self-initiated activity (\d+)d ago\) → (.+?): (.+)$')
    for i, (sub, txt) in enumerate(OFFICIAL['explanations'].items()):
        m = rx.match(txt)
        if not m:
            continue
        seg, fhs, tier, pos, neg, es, lvl, days, action, why = m.groups()
        trig = 'T1 Live collections' if action.startswith('T1') else 'T2 Just booked' if action.startswith('T2') else 'T3 Loan maturing' if action.startswith('T3') else 'None'
        rows.append(dict(SKP_CLIENT=900000001 + i, FHS=float(fhs), FHS_TIER=tier, ES=float(es), ES_LEVEL=lvl, SEGMENT=seg, SUB_SEGMENT=sub,
                         FHS_POS_REASON=pos, FHS_NEG_REASON=neg, DAYS_SINCE_LAST_SELF_ACTIVITY=float(days), NBA_ACTION=action, NBA_TRIGGER=trig,
                         EXPLANATION=txt, PERSONA=True))
    return rows


@st.cache_data(show_spinner=False)
def make_synthetic(n=10_000, seed=7):
    rng = np.random.default_rng(seed)
    subs = OFFICIAL['sub_segments']
    counts = {s: int(round(n * subs[s]['share'])) for s in SUB_ORDER}
    parts = []
    for s, k in counts.items():
        p = subs[s]
        seg = p['segment']
        if s in ('Prime Champions', 'Loyal Core', 'Sleeping Prime'):
            fhs = rng.normal(p['avg_FHS'], 9, k).clip(55, 100)
        elif s in ('Cautious Growers', 'Stretched Seekers'):
            fhs = rng.uniform(40, 54.9, k)
        elif s == 'Credit-Hungry Stressed':
            fhs = rng.normal(p['avg_FHS'], 10, k).clip(0, 39.9)
        elif s == 'Silent Risk':
            fhs = np.where(rng.random(k) < 0.8, rng.uniform(40, 54.9, k), rng.uniform(0, 39.9, k))
        else:
            fhs = rng.normal(p['avg_FHS'], 6, k).clip(10, 70)
        if s in ('Prime Champions', 'Cautious Growers', 'Stretched Seekers', 'New-to-Credit Explorers') or (s == 'Credit-Hungry Stressed'):
            es = rng.normal(max(p['avg_ES'], 42), 9, k).clip(OFFICIAL['es_cuts'][1], 95)
            if s in ('Cautious Growers', 'Credit-Hungry Stressed', 'New-to-Credit Explorers'):
                es = np.where(rng.random(k) < 0.45, rng.uniform(OFFICIAL['es_cuts'][0], OFFICIAL['es_cuts'][1] - 0.1, k), es)
        elif s == 'Loyal Core':
            es = rng.uniform(OFFICIAL['es_cuts'][0], OFFICIAL['es_cuts'][1] - 0.1, k)
        else:
            es = rng.normal(p['avg_ES'] - 3, 4, k).clip(8, OFFICIAL['es_cuts'][0] - 0.1)
            if s == 'Credit-Hungry Newcomers':
                es = rng.normal(p['avg_ES'], 8, k).clip(8, 80)
        d = pd.DataFrame({'SUB_SEGMENT': s, 'SEGMENT': seg, 'FHS': fhs.round(1), 'ES': es.round(1)})
        thin = s in ('New-to-Credit Explorers', 'Credit-Hungry Newcomers', 'Dormant Unknowns')
        d['FHS_TIER'] = 'Thin-file' if thin else d.FHS.map(_tier_from_fhs)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    k = len(df)
    df['SKP_CLIENT'] = np.arange(100001, 100001 + k)
    df['ES_LEVEL'] = df.ES.map(_level_from_es)
    df['STRESS_FLAG'] = (df.FHS_TIER.eq('Distressed') | df.SUB_SEGMENT.eq('Stretched Seekers')).astype(int)
    df['FHS_SUBGROUP'] = np.where(df.FHS_TIER.eq('Vulnerable'), np.where(df.STRESS_FLAG.eq(1), 'Vulnerable - stress signal', 'Vulnerable - building history'), df.FHS_TIER)
    df['FHS_CONFIDENCE'] = np.where(df.FHS_TIER.eq('Thin-file'), 'Low', np.where(rng.random(k) < 0.65, 'High', 'Medium'))
    df['LIVE_STRESS_FLAG'] = (rng.random(k) < 0.016).astype(int)
    pos_pool = np.array(['Past bad loans: 0', 'Loans successfully taken (track record): 5+', 'Relationship tenure: 5y+'])
    neg_pool = np.array(['Loans successfully taken (track record): 1', 'Rejections last 12m: 2+', 'Applications last 90d: 3+', 'No material negative factor'])
    df['FHS_POS_REASON'] = np.where(df.FHS_TIER.eq('Thin-file'), 'No negative record', pos_pool[rng.integers(0, 3, k)])
    df['FHS_NEG_REASON'] = np.where(df.FHS_TIER.eq('Thin-file'), 'No credit history with us (thin file)',
                                    np.where(df.FHS.ge(70), 'No material negative factor', neg_pool[rng.integers(0, 3, k)]))
    df['FHS_NEG_GROUP'] = np.where(df.FHS_NEG_REASON.str.contains('Rejections|Applications'), 'Stress signal',
                                   np.where(df.FHS_NEG_REASON.str.contains('track record|thin file'), 'Short history', 'Other'))
    for col, base in [('ES_RECENCY', 1.0), ('ES_FREQUENCY', 0.9), ('ES_INTERACTION', 0.08), ('ES_LEAD_ACTIVITY', 0.4), ('ES_PRODUCT_INTERACTION', 1.3)]:
        df[col] = (df.ES * base + rng.normal(0, 6, k)).clip(0, 100).round(1)
    df['DAYS_SINCE_LAST_SELF_ACTIVITY'] = (185 * np.exp(-df.ES / 25) + rng.uniform(0, 20, k)).clip(1, 185).round(0)
    df['MAX_OFFER'] = np.where(df.FHS_TIER.isin(['Resilient']), 12e6, np.where(df.FHS_TIER.eq('Stable'), 5e6, 2e6)) * rng.choice([0.5, 1, 1.5], k)
    df.loc[rng.random(k) < 0.12, 'MAX_OFFER'] = np.nan
    df['INCOME'] = rng.choice([4e6, 7e6, 10e6, 15e6, 25e6], k)
    df['DTI_EXISTING'] = rng.uniform(0, 0.5, k).round(3)
    df['PROVINCE'] = rng.choice(['Hồ Chí Minh', 'Hà Nội', 'Bình Dương', 'Đồng Nai', 'Cần Thơ'], k)
    # triggers (shares per segment of the final run)
    trig = np.array(['None'] * k, dtype=object)
    u = rng.random(k)
    t1 = np.select([df.SEGMENT.eq('GROW'), df.SEGMENT.eq('PROTECT')], [0.023, 0.014], 0.004)
    t2 = np.select([df.SEGMENT.eq('GROW') & ~df.SUB_SEGMENT.eq('Cautious Growers'), df.SUB_SEGMENT.isin(['Stretched Seekers', 'Credit-Hungry Stressed'])], [0.12, 0.25], 0)
    t3 = np.where(df.FHS_TIER.isin(['Resilient', 'Stable']), 0.29, 0)
    trig[u < t1] = 'T1 Live collections'
    trig[(u >= t1) & (u < t1 + t2)] = 'T2 Just booked'
    trig[(u >= t1 + t2) & (u < t1 + t2 + t3)] = 'T3 Loan maturing'
    df['NBA_TRIGGER'] = trig
    code = df.SEGMENT.map({'GROW': 'GROW_OFFER', 'REACTIVATE': 'REACTIVATE', 'PROTECT': 'PROTECT', 'BUILD': 'BUILD'}).values.astype(object)
    code[df.SUB_SEGMENT.eq('Cautious Growers').values] = 'GROW_STRUCTURED'
    for t, c in [('T3 Loan maturing', 'T3'), ('T2 Just booked', 'T2'), ('T1 Live collections', 'T1')]:
        code[df.NBA_TRIGGER.eq(t).values] = c
    df['_CODE'] = code
    df['NBA_ACTION'] = df._CODE.map(ACTION_LABEL)
    mult = df.FHS_TIER.map({'Resilient': 1.0, 'Stable': 0.8, 'Vulnerable': 0.5}).fillna(0)
    ceil = (df.MAX_OFFER.fillna(0) * mult).round(-5)
    ceil = np.where(df.FHS_TIER.eq('Thin-file'), np.minimum(0.3 * df.MAX_OFFER.fillna(3e6), 3e6).round(-5), ceil)
    ceil = np.where(df.SUB_SEGMENT.eq('New-to-Credit Explorers'), np.minimum(ceil, 1.5e6), ceil)
    ceil = np.where(df.SEGMENT.eq('PROTECT') | df._CODE.eq('T1'), 0, ceil)
    df['CREDIT_CEILING_VND'] = ceil
    credit = df._CODE.isin(['GROW_OFFER', 'GROW_STRUCTURED', 'REACTIVATE', 'T3', 'BUILD'])
    df['NBA_OFFER_VND'] = np.where(credit, df.CREDIT_CEILING_VND, 0)
    df['NBA_PRODUCT'] = np.select([df._CODE.eq('BUILD'), df._CODE.eq('T2') & df.SEGMENT.eq('PROTECT'), df._CODE.eq('T2'), credit],
                                  ['BNPL starter limit / partner-store instalment', 'Autopay / pre-due reminders (no cross-sell)',
                                   'Insurance add-on / app activation', 'Consumer-durable instalment'], '—')
    df['NBA_DETAIL'] = np.where(credit & df.NBA_OFFER_VND.le(0), 'No pre-approved limit on file: invite to apply, amount set by underwriting',
                                'See action; offer = credit ceiling scaled by FHS tier (D11)')
    high = df.ES_LEVEL.isin(['Active', 'Champion'])
    df['NBA_CHANNEL'] = np.select([df._CODE.eq('T1'), df._CODE.eq('REACTIVATE') | (df.SEGMENT.eq('REACTIVATE')), df._CODE.eq('BUILD'),
                                   df._CODE.eq('GROW_STRUCTURED'), df._CODE.isin(['GROW_OFFER', 'T3']) & high],
                                  ['Collections agent call + SMS', 'SMS / Zalo OA + in-app (no calls)', 'In-app / partner store (no outbound calls)',
                                   'In-app / SMS (no telesales push)', TEL], 'SMS / Zalo OA')
    df['NBA_TIMING'] = df._CODE.map({'T1': 'Immediately', 'T3': '60 days before last instalment', 'T2': 'Day 7 and day 30 after disbursement',
                                     'BUILD': 'Real time – when the customer arrives in-app / at a store', 'PROTECT': 'Before each due date'}).fillna('Next monthly campaign')
    growth = credit | df._CODE.eq('T2') & ~df.SEGMENT.eq('PROTECT')
    df['NBA_PRIORITY_SCORE'] = np.where(growth, 0.5 * df.ES + 0.3 * df.FHS + 20 * df._CODE.eq('T3'), 100 - df.FHS).round(1)
    df['NBA_PRIORITY'] = pd.cut(df.NBA_PRIORITY_SCORE, [-1, 40, 60, 200], labels=['Low', 'Medium', 'High']).astype(str)
    why = df._CODE.map(WHY)
    why = np.where(df._CODE.eq('PROTECT'), df.SUB_SEGMENT.map(WHY), why)
    df['EXPLANATION'] = (df.SEGMENT + ' | FHS ' + df.FHS.round(0).astype(int).astype(str) + ' (' + df.FHS_TIER + '; + ' + df.FHS_POS_REASON + '; − '
                         + df.FHS_NEG_REASON + ') | ES ' + df.ES.round(0).astype(int).astype(str) + ' (' + df.ES_LEVEL + '; last self-initiated activity '
                         + df.DAYS_SINCE_LAST_SELF_ACTIVITY.astype(int).astype(str) + 'd ago) → ' + df.NBA_ACTION + ': ' + pd.Series(why, index=df.index))
    df = df.drop(columns=['_CODE'])
    # the customers quoted in the report, so personas always exist
    pers = pd.DataFrame(_persona_rows())
    if len(pers):
        filler = df.groupby('SUB_SEGMENT').head(1).set_index('SUB_SEGMENT')
        for c in df.columns:
            if c not in pers.columns:
                pers[c] = pers.SUB_SEGMENT.map(filler[c]) if c in filler.columns else np.nan
        df = pd.concat([pers.drop(columns=['PERSONA']), df], ignore_index=True)
    return df


def _normalise(df):
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns})
    for c, default in REQUIRED.items():
        if c not in df.columns:
            df[c] = default
    df['NBA_TRIGGER'] = df['NBA_TRIGGER'].astype(str).replace({'nan': 'None', '': 'None', 'none': 'None', 'T1': 'T1 Live collections',
                                                                'T2': 'T2 Just booked', 'T3': 'T3 Loan maturing'})
    for c in ['FHS', 'ES', 'NBA_OFFER_VND', 'CREDIT_CEILING_VND', 'NBA_PRIORITY_SCORE', 'MAX_OFFER', 'DAYS_SINCE_LAST_SELF_ACTIVITY',
              'ES_RECENCY', 'ES_FREQUENCY', 'ES_INTERACTION', 'ES_LEAD_ACTIVITY', 'ES_PRODUCT_INTERACTION', 'STRESS_FLAG', 'LIVE_STRESS_FLAG']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['FHS_TIER', 'ES_LEVEL', 'SEGMENT', 'SUB_SEGMENT', 'NBA_ACTION', 'NBA_CHANNEL', 'NBA_PRIORITY', 'NBA_TIMING', 'NBA_PRODUCT', 'FHS_CONFIDENCE']:
        df[c] = df[c].astype(str).replace({'nan': ''})
    df['SKP_CLIENT'] = pd.to_numeric(df['SKP_CLIENT'], errors='coerce').fillna(-1).astype('int64')
    # weight = how many real customers one row stands for (1 for the full file; stratified samples are re-weighted to the official mix)
    if len(df) >= 999_000:
        df['WEIGHT'] = 1.0
    else:
        cnt = df.SUB_SEGMENT.value_counts()
        df['WEIGHT'] = df.SUB_SEGMENT.map(lambda s: OFFICIAL['sub_segments'].get(s, {}).get('customers', 0) / max(cnt.get(s, 1), 1)).astype(float)
        if df.WEIGHT.sum() == 0:
            df['WEIGHT'] = N_TOTAL / len(df)
    return df


def _read(path_or_buffer, name):
    if str(name).lower().endswith('.parquet'):
        return pd.read_parquet(path_or_buffer)
    comp = 'gzip' if str(name).lower().endswith('.gz') else 'infer'
    return pd.read_csv(path_or_buffer, low_memory=False, keep_default_na=False, na_values=[''], compression=comp)


@st.cache_data(show_spinner='Loading customer scores…')
def load_customer_data(uploaded_bytes=None, uploaded_name=None):
    """Find the data: upload → full parquet → full csv → sample parquet → sample csv → synthetic. Never raises."""
    if uploaded_bytes is not None:
        try:
            return _normalise(_read(io.BytesIO(uploaded_bytes), uploaded_name)), f'Uploaded file · {uploaded_name}'
        except Exception as e:
            st.sidebar.error(f'Could not read the upload ({e}); falling back to local files.')
    for fname in ['customer_scores_nba.parquet', 'customer_scores_nba.csv', 'customer_scores_nba.csv.gz', 'customer_scores_sample.parquet',
                  'customer_scores_sample.csv', 'customer_scores_sample.csv.gz']:
        for folder in [APP_DIR, os.getcwd(), os.path.join(APP_DIR, 'data')]:
            p = os.path.join(folder, fname)
            if os.path.isfile(p):
                try:
                    return _normalise(_read(p, fname)), fname
                except Exception:
                    continue
    return _normalise(make_synthetic()), 'Synthetic demo (10,000 rows)'


# =============================================================================================================
# Navigation rail + data load
# =============================================================================================================
PAGES = ['🏠  Overview', '🛡️  Financial Health', '🔥  Engagement', '🎯  Segments & NBA', '👤  Customer 360°', '💡  Insights & Decisions']
with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin:4px 0 14px 0;">'
                '<div style="background:#FFFFFF;color:#0F2043 !important;border-radius:10px;width:42px;height:42px;display:flex;'
                'align-items:center;justify-content:center;font-weight:900;font-size:1.05rem;">FT</div>'
                '<div><div style="font-weight:800;font-size:1.05rem;color:#FFFFFF !important;">CUSTOMER<br>INTELLIGENCE</div></div></div>',
                unsafe_allow_html=True)
    page = st.radio('Navigation', PAGES, label_visibility='collapsed', key='page')
    st.markdown('---')
    up = st.file_uploader('Load customer_scores_nba (.csv / .parquet)', type=['csv', 'parquet', 'gz'])
    data, data_src = load_customer_data(up.getvalue() if up is not None else None, up.name if up is not None else None)
    full_mode = len(data) >= 999_000
    mode = 'Full file' if full_mode else ('Demo data' if data_src.startswith('Synthetic') else 'Sample, re-weighted')
    st.markdown(f'<div style="font-size:.76rem;line-height:1.45;color:#CBD5E1 !important;">● <b style="color:#FFFFFF !important">{mode}</b> · '
                f'{len(data):,} rows<br>{data_src}<br>Scored as of 4 Jun 2025<br>Run {OFFICIAL["run_id"]}</div>', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<div style="font-size:.76rem;color:#CBD5E1 !important;line-height:1.4;"><b style="color:#FFFFFF !important">Team FT-US</b> · '
                'BI10 Round 2<br>Leader: Trần Long Triều<br>Home Credit · Synthesize</div>', unsafe_allow_html=True)


# =============================================================================================================
# Shared slicers, Reset-all button and the live Active-Filter banner
# =============================================================================================================
FILTER_DEFAULTS = {'f_segment': 'All', 'f_tier': 'All', 'f_level': 'All'}


def reset_all_filters():
    """One click back to the whole 1,000,528-customer book. Runs as a widget callback, i.e. before the rerun,
    so writing to the slicer keys is always legal."""
    for k_, v_ in FILTER_DEFAULTS.items():
        st.session_state[k_] = v_
    st.session_state['x_sub'] = None


def header(num, text, sub, slicers=True):
    """Page title on the left, Power BI-style slicers + Reset on the right. Slicers are shared across pages."""
    if not slicers:
        st.markdown(f'<div><span class="pg-num">{num}</span><span class="pg-title">{text}</span></div>'
                    f'<div class="pg-sub">{sub}</div>', unsafe_allow_html=True)
        return data
    cols = st.columns([2.05, 1, 1, 1, 0.78])
    with cols[0]:
        st.markdown(f'<div><span class="pg-num">{num}</span><span class="pg-title">{text}</span></div>'
                    f'<div class="pg-sub">{sub}</div>', unsafe_allow_html=True)
    seg = cols[1].selectbox('Segment', ['All'] + SEG_ORDER, key='f_segment')
    tier = cols[2].selectbox('FHS tier', ['All'] + TIER_ORDER, key='f_tier')
    lvl = cols[3].selectbox('ES level', ['All'] + LEVEL_ORDER, key='f_level')
    with cols[4]:
        st.markdown('<div style="height:25px"></div>', unsafe_allow_html=True)
        st.button('🔄 Reset All Filters', key=f'reset_{num}', on_click=reset_all_filters, **FULL,
                  help='Clear the three slicers and any chart click → back to all 1,000,528 customers')
    d = data
    if seg != 'All':
        d = d[d.SEGMENT.astype(str) == seg]
    if tier != 'All':
        d = d[d.FHS_TIER.astype(str) == tier]
    if lvl != 'All':
        d = d[d.ES_LEVEL.astype(str) == lvl]
    if st.session_state.get('x_sub'):
        d = d[d.SUB_SEGMENT.astype(str) == st.session_state['x_sub']]
    return d


def filter_banner(d):
    """The judge must always see WHY the KPIs changed: a blue live-filter bar, or a calm 'no filter' bar."""
    on = []
    for lab, k_ in [('Segment', 'f_segment'), ('FHS tier', 'f_tier'), ('ES level', 'f_level')]:
        v_ = st.session_state.get(k_, 'All')
        if v_ and v_ != 'All':
            on.append(f'{lab} = <b>{v_}</b>')
    if st.session_state.get('x_sub'):
        on.append(f'Sub-segment = <b>{st.session_state["x_sub"]}</b> <span style="font-weight:500">(from chart click)</span>')
    n = float(d.WEIGHT.sum()) if len(d) else 0.0
    if on:
        st.markdown(f'<div class="fbanner">🔍 <b>ACTIVE FILTER</b> &nbsp;→&nbsp; {" &nbsp;·&nbsp; ".join(on)}'
                    f'<br><span style="font-weight:500">Every KPI and chart on this page is recomputed live on '
                    f'≈ <b>{n:,.0f}</b> customers — {n / N_TOTAL * 100:.1f}% of the 1,000,528-customer book.</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fbanner" style="background:#F6F9F7;border-color:#CDE6D8;border-left-color:#107C41;">'
                    f'✅ <b>NO FILTER</b> &nbsp;→&nbsp; showing the full portfolio of <b>{n:,.0f}</b> customers. '
                    f'<span style="font-weight:500">Use the slicers above, or click a bar on "Customers by sub-segment".</span></div>',
                    unsafe_allow_html=True)
    gap(8)


def nba_mix_chart(d, key, height=260):
    """Next Best Action mix with SHORT y-labels; the full action sentence lives in the tooltip only."""
    n = max(float(d.WEIGHT.sum()), 1.0)
    mix = d.groupby(d.NBA_ACTION.astype(str)).WEIGHT.sum().sort_values()
    mix = mix[mix.index.astype(str) != '']
    if not len(mix):
        st.info('No customers match the current filter.')
        return
    def colmap(a):
        a = str(a)
        if a.startswith('T1'):
            return NAVY
        if a.startswith('T2'):
            return SLATE
        if a.startswith('T3'):
            return '#0E7490'
        return SEG_COLOR.get(a.split(' ')[0], NAVY)
    fig = go.Figure(go.Bar(
        y=[short_action(a) for a in mix.index], x=mix.values / n, orientation='h',
        marker_color=[colmap(a) for a in mix.index],
        text=[pct(v / n) for v in mix.values], textposition='outside', cliponaxis=False,
        textfont=dict(color=INK, size=11),
        customdata=[[str(a), v] for a, v in zip(mix.index, mix.values)],
        hovertemplate='<b>%{customdata[0]}</b><br>%{x:.1%} of the selection · %{customdata[1]:,.0f} customers<extra></extra>'))
    fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(mix.max() / n, 0.01) * 1.30])
    fig.update_yaxes(tickfont=dict(color=INK, size=11, family='Inter, Segoe UI, sans-serif'))
    show(apply_pro_theme(fig, height, False, margin=dict(l=6, r=14, t=8, b=6)), key)


# =============================================================================================================
# 01 · OVERVIEW
# =============================================================================================================
if page == PAGES[0]:
    d = header('01', 'Portfolio Overview',
               'Who is healthy, who is engaged, who is stressed – and what we do with each of the 1,000,528 customers')
    filter_banner(d)
    n = float(d.WEIGHT.sum()) if len(d) else 0.0
    seg_s = wshare(d, 'SEGMENT', SEG_ORDER)
    growth = d.NBA_ACTION.astype(str).str.startswith(('GROW', 'REACTIVATE', 'T3', 'BUILD'))
    stv = OFFICIAL['stressed']
    k = st.columns(5)
    with k[0]:
        kpi('Customers scored', f'{n:,.0f}', 'scored as of 4 Jun 2025 · one row, one action',
            pill=f'{pct(n / N_TOTAL)} of the 1,000,528 book', pill_kind='b')
    with k[1]:
        kpi('Avg Financial Health', f'{wmean(d, "FHS"):.1f}', '0–100 · 60 = average borrower · +20 points ≈ half the odds',
            GREEN, GREEN, pill='▲ 7.9× risk spread', pill_kind='g')
    with k[2]:
        kpi('Avg Engagement', f'{wmean(d, "ES"):.1f}',
            f'Cooling < {OFFICIAL["es_cuts"][0]:.0f} · Champion ≥ {OFFICIAL["es_cuts"][2]:.0f}',
            BLUE, BLUE, pill='▲ 4.6× call response', pill_kind='b')
    with k[3]:
        kpi('Financially stressed', pct(float(np.average(d.STRESS_FLAG.fillna(0), weights=d.WEIGHT)) if n else 0),
            'STRESS_FLAG = 1 · full-base rate 9.1% (91,102)', RED, RED,
            pill=f'▼ {stv["distressed"] / 1e3:.1f}k Distressed + {stv["vulnerable_with_stress_signal"] / 1e3:.1f}k Stress Vuln.',
            pill_kind='r')
    with k[4]:
        kpi('Growth actions', pct(float(np.average(growth, weights=d.WEIGHT)) if n else 0),
            'offer / re-loan / win-back / starter product', AMBER, AMBER,
            pill=f'{OFFICIAL["value_levers"]["L5_growth_actions"] / 1e3:,.0f}k growth vs '
                 f'{OFFICIAL["value_levers"]["L6_protect_actions"] / 1e3:,.0f}k protect', pill_kind='a')
    gap()
    c = st.columns([1.05, 1.05, 0.9], gap='medium')
    with c[0]:
        with card():
            title('Financial Health × Engagement', 'share of the selected customers in each FHS tier × ES level')
            tb = (d.pivot_table(index='FHS_TIER', columns='ES_LEVEL', values='WEIGHT', aggfunc='sum')
                  .reindex(index=TIER_ORDER, columns=LEVEL_ORDER).fillna(0) / max(n, 1))
            fig = go.Figure(go.Heatmap(z=tb.values, x=LEVEL_ORDER, y=TIER_ORDER,
                                       colorscale=[[0, '#F1F5F9'], [0.35, '#93C5FD'], [1, NAVY]], showscale=False,
                                       text=[[pct(v) for v in r_] for r_ in tb.values], texttemplate='%{text}',
                                       textfont=dict(size=11, family='Inter, Segoe UI, sans-serif'),
                                       hovertemplate='<b>%{y} × %{x}</b><br>%{z:.1%} of the selection<extra></extra>', xgap=3, ygap=3))
            fig.update_yaxes(autorange='reversed')
            show(apply_pro_theme(fig, 270, False), 'ov_heat')
    with c[1]:
        with card():
            title('Customers by sub-segment', '👆 click a bar to filter the whole page')
            sub = d.groupby(d.SUB_SEGMENT.astype(str)).WEIGHT.sum().reindex(SUB_ORDER).fillna(0)
            order = SUB_ORDER[::-1]
            fig = go.Figure(go.Bar(
                y=order, x=sub.reindex(order).values, orientation='h',
                marker_color=[SEG_COLOR[OFFICIAL['sub_segments'][s]['segment']] for s in order],
                text=[f'{v / 1e3:,.0f}k' for v in sub.reindex(order).values], textposition='outside', cliponaxis=False,
                textfont=dict(color=INK, size=11),
                customdata=[OFFICIAL['sub_segments'][s]['segment'] for s in order],
                hovertemplate='<b>%{y}</b> (%{customdata})<br>%{x:,.0f} customers<br><i>click to filter the page</i><extra></extra>'))
            fig.update_xaxes(showticklabels=False, showgrid=False, range=[0, max(float(sub.max()), 1) * 1.22])
            fig.update_yaxes(tickfont=dict(color=INK, size=11, family='Inter, Segoe UI, sans-serif'))
            ev = show(apply_pro_theme(fig, 270, False), 'ov_sub', select=True)
            pts = chart_points(ev)                      # never raises, on any Streamlit version
            if pts:
                lbl = point_field(pts[0], 'y')
                if lbl in SUB_ORDER and st.session_state.get('x_sub') != lbl:
                    st.session_state['x_sub'] = lbl
                    st.rerun()
    with c[2]:
        with card():
            title('Four segments', 'one primary action each')
            fig = go.Figure(go.Pie(
                labels=SEG_ORDER, values=seg_s.values, hole=0.62,
                marker=dict(colors=[SEG_COLOR[s] for s in SEG_ORDER], line=dict(color='#FFFFFF', width=2)),
                textinfo='percent', textfont=dict(color='#FFFFFF', size=12, family='Inter, Segoe UI, sans-serif'),
                sort=False, direction='clockwise',
                hovertemplate='<b>%{label}</b><br>%{percent} of the selection<extra></extra>'))
            fig.add_annotation(text=(f'<b>{n / 1e6:,.2f}M</b>' if n >= 1e6 else f'<b>{n / 1e3:,.0f}k</b>') + '<br>customers',
                               showarrow=False, font=dict(size=13, color=INK))
            show(apply_pro_theme(fig, 270, True), 'ov_donut')
    gap()
    c = st.columns([1.05, 1.05, 0.9], gap='medium')
    with c[0]:
        with card():
            title('FHS distribution', 'customers per 5-point band, coloured by tier')
            fig = go.Figure()
            for t in TIER_ORDER:
                dd = d[d.FHS_TIER.astype(str) == t]
                if len(dd):
                    fig.add_trace(go.Histogram(x=dd.FHS, y=dd.WEIGHT, histfunc='sum',
                                               xbins=dict(start=0, end=100, size=5), name=t, marker_color=TIER_COLOR[t],
                                               hovertemplate='<b>' + t + '</b> · FHS %{x}<br>%{y:,.0f} customers<extra></extra>'))
            fig.update_layout(barmode='stack')
            fig.update_xaxes(title=dict(text='Financial Health Score'))
            show(apply_pro_theme(fig, 260, True, bargap=0.08), 'ov_hist')
    with c[1]:
        with card():
            title('Next Best Action mix', 'triggers T1 → T2 → T3 override the segment action')
            nba_mix_chart(d, 'ov_nba', 260)
    with c[2]:
        takeaway([('Risk is concentrated',
                   'Bad rate on new loans falls <b>7.9×</b> from Distressed (4.40%) to Resilient (0.56%) – growth does not have to be cautious everywhere.'),
                  ('Vulnerable ≠ stressed',
                   '87% of Vulnerable (200,042) only lack history; the real stress pool is 91,102 customers (9.1%).'),
                  ('The eager borrower is not the safe one',
                   'Engaged thin-file customers answer calls most (1.70 / 100) but their first loans go bad at 11.9%.'),
                  ('One action per customer',
                   'Every row of the output file carries product, offer, channel, timing and a plain-English reason.')])
    source(SRC + ' · every chart on this page recomputes from the loaded file and the slicers')

# =============================================================================================================
# 02 · FINANCIAL HEALTH (Challenge 1)
# =============================================================================================================
elif page == PAGES[1]:
    d = header('02', 'Financial Health Score', 'C1 · Which customers can carry more credit – and which are stressed?')
    filter_banner(d)
    stv = OFFICIAL['stressed']
    k = st.columns(4)
    with k[0]:
        kpi('Out-of-time AUC', '0.661', '838,391 loans the fit never saw', GREEN, GREEN,
            pill='explainable · black box 0.696', pill_kind='g')
    with k[1]:
        kpi('Bad-rate spread', '7.9×', 'Distressed 4.40% → Resilient 0.56% (back-test)',
            pill='▲ 13× on one past bad loan', pill_kind='b')
    with k[2]:
        kpi('Financially stressed', f"{stv['stressed_total']:,}", f"{pct(stv['stressed_share'])} of the book · STRESS_FLAG = 1",
            RED, RED, pill=f"▼ {stv['distressed']:,} Distressed + {stv['vulnerable_with_stress_signal']:,} Stress Vuln.", pill_kind='r')
    with k[3]:
        kpi('Vulnerable – building history', f"{stv['vulnerable_short_history_clean']:,}", '87% of Vulnerable carry no stress signal',
            AMBER, AMBER, pill='structure, do not decline', pill_kind='a')
    gap()
    c = st.columns([1, 1.25], gap='medium')
    with c[0]:
        with card():
            title('What predicts financial distress?', 'Information Value of 18 candidate factors · behaviour beats demographics')
            IVN = {'prior_bad': ('Past bad loans', 0), 'max_offer': ("HC's own limit (MAX_OFFER)", 2), 'prior_booked': ('Loans taken (track record)', 0),
                   'NAME_EDUCATION_TYPE': ('Education', 3), 'tenure_days': ('Relationship tenure', 0), 'DOWNPAY_RATIO': ('Down-payment ratio', 1),
                   'PAYMENT_NUM': ('Tenor', 1), 'prior_severe': ('Past write-off / sold', 0), 'FLAG_INS': ('Insurance on loan', 1),
                   'apps_90': ('Applications 90d', 0), 'rej_365': ('Rejections 12m', 0), 'PTI': ('Payment-to-income', 1),
                   'NAME_GENDER': ('Gender', 3), 'scoring_channels_90': ('Channels shopped 90d', 0), 'AGE_AT_APP': ('Age', 3),
                   'n_active_inst': ('Active instalment loans', 4), 'dti_existing': ('Debt-to-income', 4), 'income': ('Declared income', 0)}
            CAT = [('Behavioural · in FHS', NAVY), ('Loan-design lever', AMBER), ("HC's limit · validation", PURPLE),
                   ('Demographic · excluded', '#7C8DA4'), ('Tested · dropped', '#A8B4C4')]
            iv = sorted(OFFICIAL['iv'].items(), key=lambda kv: kv[1])
            fig = go.Figure()
            for ci, (cn, col) in enumerate(CAT):
                sel = [(IVN[k_][0], v) for k_, v in iv if IVN[k_][1] == ci]
                if sel:
                    fig.add_trace(go.Bar(y=[s_[0] for s_ in sel], x=[s_[1] for s_ in sel], orientation='h', name=cn, marker_color=col,
                                         text=[f'{s_[1]:.3f}' for s_ in sel], textposition='outside', cliponaxis=False,
                                         textfont=dict(color=INK, size=10),
                                         hovertemplate='<b>%{y}</b><br>Information Value %{x:.3f}<extra></extra>'))
            fig.update_layout(yaxis=dict(categoryorder='array', categoryarray=[IVN[k_][0] for k_, _ in iv]))
            fig.update_xaxes(range=[0, 0.25], showticklabels=False)
            fig.update_yaxes(tickfont=dict(color=INK, size=10, family='Inter, Segoe UI, sans-serif'))
            show(apply_pro_theme(fig, 420), 'fh_iv')
            source('Figure A2 · 535,487 training loans · DTI and income ≈ 0.004 because underwriting already filters affordability')
    with c[1]:
        with card():
            title('Five tiers, one clear risk ladder', 'selected customers · bad rate on new loans · Home Credit’s own median limit')
            share = wshare(d, 'FHS_TIER', TIER_ORDER)
            tr = TIER_ORDER[::-1]
            fig = make_subplots(rows=1, cols=3, subplot_titles=('Share of customers', 'Bad rate, new loans', 'Median MAX_OFFER'),
                                horizontal_spacing=0.1)
            fig.add_trace(go.Bar(y=tr, x=[share[t] for t in tr], orientation='h', marker_color=[TIER_COLOR[t] for t in tr],
                                 text=[pct(share[t]) for t in tr], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{y}</b><br>%{x:.1%} of the selection<extra></extra>'), 1, 1)
            fig.add_trace(go.Bar(y=tr, x=[OFFICIAL['tier_bad_rate'][t] for t in tr], orientation='h',
                                 marker_color=[TIER_COLOR[t] for t in tr],
                                 text=[pct(OFFICIAL['tier_bad_rate'][t], 2) for t in tr], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{y}</b><br>bad rate %{x:.2%}<extra></extra>'), 1, 2)
            mo = [t for t in tr if t in OFFICIAL['tier_median_offer']]
            fig.add_trace(go.Bar(y=mo, x=[OFFICIAL['tier_median_offer'][t] for t in mo], orientation='h',
                                 marker_color=[TIER_COLOR[t] for t in mo],
                                 text=[f"{OFFICIAL['tier_median_offer'][t] / 1e6:.0f}M" for t in mo], textposition='outside',
                                 cliponaxis=False, textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{y}</b><br>median limit %{x:,.0f} VND<extra></extra>'), 1, 3)
            fig.update_xaxes(showticklabels=False, showgrid=False)
            fig.update_yaxes(showticklabels=False, col=2)
            fig.update_yaxes(showticklabels=False, col=3)
            apply_pro_theme(fig, 245, False, margin=dict(l=6, r=26, t=28, b=6))
            fig.update_yaxes(tickfont=dict(color=INK, size=11, family='Inter, Segoe UI, sans-serif'), col=1)
            fig.update_annotations(font=dict(size=11, color=INK))
            show(fig, 'fh_tiers')
            dbt = OFFICIAL['drivers_by_tier']
            order = ['Distressed', 'Vulnerable', 'Stable', 'Resilient']
            title('Main negative reason per tier', '"Vulnerable" is mostly short history, not stress')
            fig = go.Figure()
            for g, col, nm in [('Stress signal', RED, 'Stress signal'), ('Short history', AMBER, 'Short history'),
                               ('Other', '#A8B4C4', 'No material negative')]:
                fig.add_trace(go.Bar(y=order, x=[dbt[t][g] for t in order], orientation='h', name=nm,
                                     marker=dict(color=col, line=dict(color='#FFFFFF', width=2)),
                                     text=[pct(dbt[t][g], 0) if dbt[t][g] > 0.05 else '' for t in order], textposition='inside',
                                     insidetextfont=dict(color='#FFFFFF', size=10),
                                     hovertemplate='<b>%{y}</b> · ' + nm + ': %{x:.1%}<extra></extra>'))
            fig.update_layout(barmode='stack')
            fig.update_xaxes(tickformat='.0%')
            fig.update_yaxes(tickfont=dict(color=INK, size=11, family='Inter, Segoe UI, sans-serif'))
            show(apply_pro_theme(fig, 180), 'fh_drv_tier')
    gap()
    # ---------------------------------------------------------------------------------------------------------
    # Interactive FHS Scorecard Simulator (What-If Calculator)
    # ---------------------------------------------------------------------------------------------------------
    c = st.columns([1.32, 1, 0.72], gap='medium')
    with c[0]:
        with card():
            title('🧮 Interactive FHS Scorecard Simulator (What-If)',
                  'build any customer profile → points added to / taken from the 60-point base, live')
            SC = OFFICIAL['scorecard']
            BR = OFFICIAL['scorecard_badrate']
            FACT = [('Past late payments', 'Past bad loans', ['0', '1', '2+'], '0'),
                    ('Past write-off / sold debt', 'Write-off', ['0', '1+'], '0'),
                    ('Loans successfully taken (track record)', 'Loans taken', ['0', '1', '2', '3-4', '5+'], '1'),
                    ('Relationship tenure', 'Tenure', ['No history', '<1y', '1-3y', '3-5y', '5y+'], '<1y'),
                    ('Applications last 90d', 'Applications 90d', ['0', '1', '2', '3+'], '0'),
                    ('Rejections last 12m', 'Rejections 12m', ['0', '1', '2+'], '0'),
                    ('Channels shopped last 90d (credit hunger)', 'Channels 90d', ['0', '1', '2', '3+'], '0'),
                    ('Declared monthly income', 'Income', ['<5M', '5-8M', '8-12M', '12-20M', '>20M'], '8-12M')]
            cc = st.columns(4)
            choice = {}
            for i, (f, lab, opts, dflt) in enumerate(FACT):
                choice[f] = cc[i % 4].selectbox(lab, opts, index=opts.index(dflt), key=f'fh_calc_{i}')
            pts = {lab: SC[f][choice[f]] for f, lab, _, _ in FACT}
            raw = 60 + sum(pts.values())
            fhs_c = float(np.clip(raw, 0, 100))
            capped = choice['Past write-off / sold debt'] == '1+' and fhs_c > 30
            fhs_c = 30.0 if capped else fhs_c
            thin = choice['Loans successfully taken (track record)'] == '0'
            tier_c = ('Thin-file' if thin else
                      ('Resilient' if fhs_c >= 70 else 'Stable' if fhs_c >= 55 else 'Vulnerable' if fhs_c >= 40 else 'Distressed'))
            fig = go.Figure(go.Waterfall(
                measure=['absolute'] + ['relative'] * len(pts) + ['total'],
                x=['Base'] + list(pts) + ['FHS'], y=[60] + list(pts.values()) + [0],
                text=['60'] + [f'{v:+.1f}' for v in pts.values()] + [f'{fhs_c:.1f}'], textposition='outside',
                cliponaxis=False, textfont=dict(color=INK, size=10),
                connector=dict(line=dict(color=ZERO_C)),
                increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
                totals=dict(marker=dict(color=TIER_COLOR[tier_c])),
                hovertemplate='<b>%{x}</b><br>%{y:+.2f} points<extra></extra>'))
            for yv, nm in ((40, 'Vulnerable ≥ 40'), (55, 'Stable ≥ 55'), (70, 'Resilient ≥ 70')):
                fig.add_hline(y=yv, line=dict(color=ZERO_C, dash='dot'),
                              annotation_text=nm, annotation_position='right',
                              annotation_font=dict(size=9, color=INK_SOFT))
            fig.update_yaxes(range=[min(0, raw - 10), max(100, raw + 10)], title=dict(text='FHS points'))
            fig.update_xaxes(tickfont=dict(color=INK_TICK, size=10, family='Inter, Segoe UI, sans-serif'))
            show(apply_pro_theme(fig, 255, False, margin=dict(l=6, r=70, t=10, b=6)), 'fh_wf')
            # --- the resulting tier + the reason code the engine would write -------------------------------
            worst = min(pts, key=lambda kk: pts[kk])
            best = max(pts, key=lambda kk: pts[kk])
            lab2fact = {lab: f for f, lab, _, _ in FACT}
            rc_neg = ('No credit history with us (thin file)' if thin else
                      (f'{worst}: {choice[lab2fact[worst]]}' if pts[worst] < 0 else 'No material negative factor'))
            rc_pos = 'No negative record' if thin else f'{best}: {choice[lab2fact[best]]}'
            br_key = lab2fact[worst]
            obs_bad = BR[br_key][choice[br_key]]
            cap_note = ' <span class="pill pill-r">capped at 30 · write-off</span>' if capped else ''
            thin_note = ' <span class="pill pill-s">no disbursed loan → Thin-file</span>' if thin else ''
            st.markdown(
                f'<div style="text-align:center;margin-top:4px;">'
                f'<span style="color:{INK_SOFT};font-size:.86rem;font-weight:600">FHS </span>'
                f'<b style="font-size:1.5rem;color:{TIER_COLOR[tier_c]}">{fhs_c:.1f}</b>'
                f'&nbsp;<span class="tag" style="background:{TIER_COLOR[tier_c]}">{tier_c}</span>'
                f'{cap_note}{thin_note}</div>'
                f'<div class="rule" style="border-left-color:{TIER_COLOR[tier_c]};margin-top:8px;">'
                f'<b>REASON CODE the engine writes</b><br>➕ strongest positive: {rc_pos}<br>➖ strongest negative: {rc_neg}'
                f'<br><span style="color:{INK_SOFT}">Observed bad rate of the driving band: <b>{pct(obs_bad, 2)}</b></span></div>',
                unsafe_allow_html=True)
            source('Report worked example: never late, 1 loan, < 1 year → 54.6 (Vulnerable); tenure 1–3y → 57.1 (Stable)')
    with c[1]:
        with card():
            title('🎚️ Interactive Credit-Appetite Cut-off', 'drag the approval threshold → volume kept vs bad loans avoided')
            CUT_LABELS = {'All (no cut-off)': 0, 'FHS ≥ 40': 40, 'FHS ≥ 55': 55, 'FHS ≥ 60': 60, 'FHS ≥ 80': 80}
            lab = st.select_slider('Approve customers at', options=list(CUT_LABELS.keys()), value='FHS ≥ 40', key='fh_cut')
            cut = CUT_LABELS[lab]
            rl, rv = OFFICIAL['tradeoff_loans'][cut], OFFICIAL['tradeoff_vnd'][cut]
            st.markdown(f'<div style="color:{INK};font-weight:700;font-size:.80rem;margin:2px 0 3px 0">BY NUMBER OF LOANS</div>',
                        unsafe_allow_html=True)
            m = st.columns(3)
            stat(m[0], '% Volume kept', pct(rl['approved']), GREEN)
            stat(m[1], 'Bad rate', pct(rl['bad_rate'], 2))
            stat(m[2], '% Bad avoided', pct(rl['avoided']), RED)
            st.markdown(f'<div style="color:{INK};font-weight:700;font-size:.80rem;margin:8px 0 3px 0">BY CREDIT AMOUNT (VND)</div>',
                        unsafe_allow_html=True)
            m = st.columns(3)
            stat(m[0], '% Amount kept', pct(rv['approved']), GREEN)
            stat(m[1], 'Bad rate', pct(rv['bad_rate'], 2))
            stat(m[2], '% Bad avoided', pct(rv['avoided']), RED)
            cuts_v = sorted(OFFICIAL['tradeoff_vnd'].keys())
            cuts_l = sorted(OFFICIAL['tradeoff_loans'].keys())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[OFFICIAL['tradeoff_loans'][k_]['approved'] for k_ in cuts_l],
                                     y=[OFFICIAL['tradeoff_loans'][k_]['bad_rate'] for k_ in cuts_l],
                                     mode='lines+markers', name='by loans', line=dict(color=NAVY, width=2),
                                     marker=dict(size=6, color=NAVY), customdata=cuts_l,
                                     hovertemplate='FHS ≥ %{customdata}<br>keep %{x:.1%} of loans · bad %{y:.2%}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[OFFICIAL['tradeoff_vnd'][k_]['approved'] for k_ in cuts_v],
                                     y=[OFFICIAL['tradeoff_vnd'][k_]['bad_rate'] for k_ in cuts_v],
                                     mode='lines+markers', name='by VND', line=dict(color=PURPLE, width=2, dash='dot'),
                                     marker=dict(size=6, color=PURPLE), customdata=cuts_v,
                                     hovertemplate='FHS ≥ %{customdata}<br>keep %{x:.1%} of VND · bad %{y:.2%}<extra></extra>'))
            fig.add_trace(go.Scatter(x=[rl['approved']], y=[rl['bad_rate']], mode='markers', showlegend=False,
                                     marker=dict(size=15, color=RED, line=dict(color='#FFFFFF', width=2)), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[rv['approved']], y=[rv['bad_rate']], mode='markers', showlegend=False,
                                     marker=dict(size=13, color=RED, symbol='diamond', line=dict(color='#FFFFFF', width=2)),
                                     hoverinfo='skip'))
            fig.update_xaxes(tickformat='.0%', title=dict(text='volume kept'))
            fig.update_yaxes(tickformat='.1%', title=dict(text='bad rate'))
            show(apply_pro_theme(fig, 215, True, margin=dict(l=6, r=12, t=22, b=6)), 'fh_sim')
            source('Figure A7b · 838,391 out-of-time loans')
    with c[2]:
        takeaway([('Behaviour beats declarations',
                   'One past bad loan: 0.46% → 6.15% (<b>13×</b>). Income and DTI ≈ 0 – underwriting already filters them.'),
                  ('Cheapest protection', 'Decline FHS < 40: give up 3.4% of loans, avoid <b>10.6%</b> of bad loans.'),
                  ('Structure, don’t decline', 'FHS 40–60 is 49% of loans at 1.98%: smaller ticket, ≤ 12 m, down payment ≥ 30%.'),
                  ('Explainable & fair', 'No demographic input; mean FHS women 64.5 vs men 64.2. Recalibrate levels quarterly.')])
    source(SRC)

# =============================================================================================================
# 03 · ENGAGEMENT (Challenge 2)
# =============================================================================================================
elif page == PAGES[2]:
    d = header('03', 'Engagement Score', 'C2 · Who is listening? Measured only on what the customer initiates')
    filter_banner(d)
    evv = OFFICIAL['es_validation']
    fw = OFFICIAL['es_forward']
    pr = OFFICIAL['es_profile']
    i1 = OFFICIAL['i1']
    k = st.columns(4)
    with k[0]:
        kpi('Hold-out AUC', f"{evv['Forward HOLD-OUT half (not used for weights): positive response']['auc']:.3f}",
            'positive answer to a sales call · 170,368 customers', BLUE, BLUE, pill='weights never saw this half', pill_kind='b')
    with k[1]:
        kpi('Inbound AUC', f"{evv['Forward: customer calls us inbound']['auc']:.3f}", 'customer calls us · all 1,000,528',
            pill='▲ strongest validation', pill_kind='g')
    with k[2]:
        kpi('Champion vs Cooling', f"{fw['Champion']['positive_per_100_calls'] / fw['Cooling']['positive_per_100_calls']:.1f}×",
            'positive answers per sales call', GREEN, GREEN, pill='1.38 vs 0.30 per 100 calls', pill_kind='g')
    with k[3]:
        kpi('Calls freed', f"{OFFICIAL['value_levers']['L1_calls_saved_cut30pct_lowES'] / 1e3:,.0f}k",
            'cut the 30% lowest-ES calls', RED, RED,
            pill=f"keep {pct(OFFICIAL['value_levers']['L1_positives_kept'], 0)} of positives", pill_kind='r')
    gap()
    c = st.columns([1, 1.15, 0.8], gap='medium')
    with c[0]:
        with card():
            title('Five dimensions by level', 'R 20% · F 30% · I 15% · L 10% · P 25% · half-life 180 days')
            fig = go.Figure()
            for lv in LEVEL_ORDER:
                fig.add_trace(go.Bar(x=['Recency', 'Frequency', 'Interaction', 'Lead', 'Product'],
                                     y=[pr[lv][k_] for k_ in 'RFILP'], name=lv,
                                     marker=dict(color=LEVEL_COLOR[lv], line=dict(color='#FFFFFF', width=1)),
                                     hovertemplate='<b>' + lv + '</b> · %{x}<br>score %{y:.1f}<extra></extra>'))
            fig.update_layout(barmode='group')
            show(apply_pro_theme(fig, 300, True, bargap=0.2), 'es_rfilp')
            f96 = pr['Champion']['cie_12m'] / pr['Cooling']['cie_12m']
            p19 = pr['Champion']['n_push_leads'] / pr['Cooling']['n_push_leads']
            source(f'Customer-initiated events rise {f96:.1f}× from Cooling to Champion; company push only {p19:.1f}× '
                   f'and the connect rate is flat')
    with c[1]:
        with card():
            title('What drives – and what reduces – engagement', 'lift = outcome rate with ÷ without the signal (log scale)')
            drv = pd.DataFrame(OFFICIAL['es_drivers']).sort_values('lift')
            cols = [GREEN if str(f).startswith('+') else RED if str(f).startswith('−') else '#7C8DA4' for f in drv.factor]
            fig = go.Figure(go.Bar(
                y=[str(f)[:46] for f in drv.factor], x=drv.lift, orientation='h', marker_color=cols,
                text=[f'×{v:.2f}' for v in drv.lift], textposition='outside', cliponaxis=False,
                textfont=dict(color=INK, size=10),
                customdata=np.stack([drv.outcome.to_numpy(), drv.share.to_numpy()], axis=-1),
                hovertemplate='<b>%{y}</b><br>×%{x:.2f} on %{customdata[0]}<br>present in %{customdata[1]:.1%} of customers<extra></extra>'))
            fig.add_vline(x=1, line=dict(color=INK_SOFT, width=1))
            fig.update_xaxes(type='log', range=[np.log10(0.25), np.log10(45)], showticklabels=False)
            fig.update_yaxes(tickfont=dict(color=INK, size=10, family='Inter, Segoe UI, sans-serif'))
            show(apply_pro_theme(fig, 330, False, margin=dict(l=6, r=42, t=8, b=6)), 'es_drv')
    with c[2]:
        with card():
            title('ES distribution', 'selected customers')
            fig = go.Figure()
            for lv in LEVEL_ORDER:
                dd = d[d.ES_LEVEL.astype(str) == lv]
                if len(dd):
                    fig.add_trace(go.Histogram(x=dd.ES, y=dd.WEIGHT, histfunc='sum', xbins=dict(start=0, end=100, size=4),
                                               name=lv, marker_color=LEVEL_COLOR[lv],
                                               hovertemplate='<b>' + lv + '</b> · ES %{x}<br>%{y:,.0f} customers<extra></extra>'))
            fig.update_layout(barmode='stack')
            fig.update_xaxes(title=dict(text='Engagement Score'))
            show(apply_pro_theme(fig, 300, True, bargap=0.05), 'es_hist')
    gap()
    c = st.columns([1, 1.1, 1.05], gap='medium')
    with c[0]:
        with card():
            title('PULL beats PUSH, 3 to 1', 'who started the conversation decides whether it converts')
            order_i1 = ['PULL', 'WARM', 'PUSH']
            ICOL = {'PULL': GREEN, 'WARM': AMBER, 'PUSH': RED}
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Books within 30 days', 'Share of all leads'),
                                horizontal_spacing=0.18)
            fig.add_trace(go.Bar(x=order_i1, y=[i1[s]['book_30d'] for s in order_i1],
                                 marker_color=[ICOL[s] for s in order_i1],
                                 text=[pct(i1[s]['book_30d']) for s in order_i1], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=11),
                                 customdata=[i1[s]['leads'] for s in order_i1],
                                 hovertemplate='<b>%{x}</b><br>books %{y:.1%} within 30d<br>%{customdata:,.0f} leads<extra></extra>'), 1, 1)
            fig.add_trace(go.Bar(x=order_i1, y=[i1[s]['lead_share'] for s in order_i1],
                                 marker_color=[ICOL[s] for s in order_i1],
                                 text=[pct(i1[s]['lead_share']) for s in order_i1], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=11),
                                 hovertemplate='<b>%{x}</b><br>%{y:.1%} of all leads<extra></extra>'), 1, 2)
            fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1.12])
            apply_pro_theme(fig, 265, False, margin=dict(l=6, r=12, t=30, b=6))
            fig.update_annotations(font=dict(size=11, color=INK))
            show(fig, 'es_pullpush')
            source(f"PULL converts {pct(i1['PULL']['book_30d'])} but is only {pct(i1['PULL']['lead_share'])} of leads · "
                   f"PUSH converts {pct(i1['PUSH']['book_30d'])} on {pct(i1['PUSH']['lead_share'], 0)} of the volume")
    with c[1]:
        with card():
            title('📞 Telesales Call-Budget Optimizer', 'cut the lowest-ES calls and watch the positives you keep')
            cut_pct = st.slider('Cut % of the lowest-ES sales calls', 0, 60, 30, 5, key='es_cut',
                                help='Default 30% = the scenario quoted to the Board: ~821k calls freed, 87% of positives kept')
            tot = OFFICIAL['value_levers']['forward_sales_attempts_total']
            keep_share = 1 - cut_pct / 100
            keep = gains(keep_share) if cut_pct else 1.0
            m = st.columns(3)
            stat(m[0], 'Calls saved', f'{tot * cut_pct / 100 / 1e3:,.0f}k', RED)
            stat(m[1], 'Positives kept', pct(keep, 0), GREEN)
            stat(m[2], 'Per-call lift', f'+{(keep / max(keep_share, 1e-9) - 1) * 100:.0f}%' if cut_pct else '+0%', BLUE)
            gx = np.linspace(0, 1, 101)
            fig = go.Figure(go.Scatter(x=gx, y=monotone_curve(GAINS_X, GAINS_Y, gx), mode='lines',
                                       line=dict(color=BLUE, width=2.5), name='ES-ranked calls',
                                       hovertemplate='keep the top %{x:.0%} of calls → %{y:.0%} of positives<extra></extra>'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#A8B4C4', dash='dash'),
                                     name='random calling', hoverinfo='skip'))
            if cut_pct:
                # the two guide lines the judge asked for: drop down and across to the operating point
                fig.add_shape(type='line', x0=keep_share, x1=keep_share, y0=0, y1=keep,
                              line=dict(color=RED, width=1.5, dash='dot'))
                fig.add_shape(type='line', x0=0, x1=keep_share, y0=keep, y1=keep,
                              line=dict(color=RED, width=1.5, dash='dot'))
                fig.add_trace(go.Scatter(x=[keep_share], y=[keep], mode='markers', showlegend=False,
                                         marker=dict(size=14, color=RED, line=dict(color='#FFFFFF', width=2)),
                                         hovertemplate=f'cut {cut_pct}% of calls<br>keep {keep:.1%} of positives<extra></extra>'))
                fig.add_annotation(x=keep_share, y=keep, text=f'<b>{keep:.0%}</b> of positives<br>on {keep_share:.0%} of calls',
                                   showarrow=True, arrowhead=0, ax=-58, ay=34, font=dict(size=10, color=INK),
                                   bgcolor='rgba(255,255,255,.88)', bordercolor=RED, borderwidth=1, borderpad=3)
            fig.update_xaxes(tickformat='.0%', title=dict(text='share of sales calls kept'))
            fig.update_yaxes(tickformat='.0%', title=dict(text='positives kept'))
            show(apply_pro_theme(fig, 215, True, margin=dict(l=6, r=14, t=24, b=6)), 'es_gains')
            source('Measured at 20% and 70% of calls; monotone interpolation in between')
    with c[2]:
        with card():
            title('Engaged customers answer, call and stay', 'outcome by level · forward results are associations')
            bt = OFFICIAL['es_backtest_active']
            fig = make_subplots(rows=1, cols=3, subplot_titles=('Still self-active', 'Positives / 100 calls', 'Calls us inbound'),
                                horizontal_spacing=0.08)
            fig.add_trace(go.Bar(x=LEVEL_ORDER, y=[bt[l] for l in LEVEL_ORDER],
                                 marker_color=[LEVEL_COLOR[l] for l in LEVEL_ORDER],
                                 text=[pct(bt[l], 0) for l in LEVEL_ORDER], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{x}</b><br>stays self-active %{y:.1%}<extra></extra>'), 1, 1)
            fig.add_trace(go.Bar(x=LEVEL_ORDER, y=[fw[l]['positive_per_100_calls'] for l in LEVEL_ORDER],
                                 marker_color=[LEVEL_COLOR[l] for l in LEVEL_ORDER],
                                 text=[f"{fw[l]['positive_per_100_calls']:.2f}" for l in LEVEL_ORDER], textposition='outside',
                                 cliponaxis=False, textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{x}</b><br>%{y:.2f} positives / 100 calls<extra></extra>'), 1, 2)
            fig.add_trace(go.Bar(x=LEVEL_ORDER, y=[fw[l]['inbound_rate'] for l in LEVEL_ORDER],
                                 marker_color=[LEVEL_COLOR[l] for l in LEVEL_ORDER],
                                 text=[pct(fw[l]['inbound_rate']) for l in LEVEL_ORDER], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{x}</b><br>calls us inbound %{y:.1%}<extra></extra>'), 1, 3)
            fig.update_yaxes(showticklabels=False, showgrid=False)
            apply_pro_theme(fig, 265, False, margin=dict(l=6, r=12, t=30, b=6))
            fig.update_annotations(font=dict(size=10, color=INK))
            show(fig, 'es_val')
    gap()
    c = st.columns([1, 1], gap='medium')
    with c[0]:
        with card():
            title('Validation on four independent outcomes', 'the score was fitted on none of them')
            vt = pd.DataFrame([[k_, f"{v['n']:,}", pct(v['rate'], 2), f"{v['auc']:.4f}"] for k_, v in evv.items()],
                              columns=['Outcome', 'Customers', 'Base rate', 'AUC'])
            table_html(vt)
    with c[1]:
        takeaway([('Measure the customer, not the company', '94% of leads are pushed; only customer-initiated events count.'),
                  ('Past initiative drives', 'A past positive answer: <b>20.3×</b> · inbound call: 3.6× · ≥ 2 product families: 2.4×.'),
                  ('Pushing backfires', 'Declined pushes 0.65× · ≥ 5 unanswered calls 0.81× → cap at 5 attempts.'),
                  ('The borrowing clock', 'Just borrowed → 22% act again vs 38% a month later: timing via T2 / T3, not ES.')])
    source(SRC)

# =============================================================================================================
# 04 · SEGMENTS & NBA (Challenge 3)
# =============================================================================================================
elif page == PAGES[3]:
    d = header('04', 'Segments & Next Best Action',
               'C3 · Is it safe to lend more? Is the customer listening? → one action per customer')
    filter_banner(d)
    seg_s = wshare(d, 'SEGMENT', SEG_ORDER)
    k = st.columns(4)
    for i, s in enumerate(SEG_ORDER):
        with k[i]:
            o = OFFICIAL['segments'][s]
            kpi(s, pct(seg_s[s]), f"{o['customers']:,} customers in the full base", SEG_COLOR[s], SEG_COLOR[s],
                pill=f"bad {pct(o['bt_newloan_bad'])} · {o['fwd_positive_per_100_calls']:.2f} pos/100 calls",
                pill_kind=('g' if s == 'GROW' else 'r' if s == 'PROTECT' else 'b' if s == 'REACTIVATE' else 'a'))
    gap()
    c = st.columns([1.05, 1.15], gap='medium')
    with c[0]:
        with card():
            title('FHS tier × ES band → 4 segments', 'with the two risk guard-rails (⛔)')
            ms = OFFICIAL['matrix_share']
            def cell(seg, sub, share, x=''):
                return (f'<td style="background:{SEG_COLOR[seg]}"><div>{seg}</div>'
                        f'<div style="font-weight:500;opacity:.94">{sub}</div>'
                        f'<div style="font-weight:500;opacity:.94">{pct(share)}{x}</div></td>')
            st.markdown(f"""<table class="grid" style="width:100%;border-collapse:separate;">
<tr><th></th><th>High (Active + Champion)</th><th>Medium (Warm)</th><th>Low (Cooling)</th></tr>
<tr><td class="rowh">Healthy</td>{cell('GROW', 'Prime Champions', ms['Healthy (Resilient + Stable)']['High'])}{cell('GROW', 'Loyal Core', ms['Healthy (Resilient + Stable)']['Medium'])}{cell('REACTIVATE', 'Sleeping Prime', ms['Healthy (Resilient + Stable)']['Low'])}</tr>
<tr><td class="rowh">Vulnerable</td>{cell('GROW', 'Cautious Growers', ms['Vulnerable']['High'], ' · ⛔ stress → PROTECT')}{cell('GROW', 'Cautious Growers', ms['Vulnerable']['Medium'], ' · ⛔ stress → PROTECT')}{cell('PROTECT', 'Silent Risk', ms['Vulnerable']['Low'])}</tr>
<tr><td class="rowh">Distressed</td>{cell('PROTECT', 'Credit-Hungry Stressed', ms['Distressed']['High'])}{cell('PROTECT', 'Credit-Hungry Stressed', ms['Distressed']['Medium'])}{cell('PROTECT', 'Silent Risk', ms['Distressed']['Low'])}</tr>
<tr><td class="rowh">Thin-file</td>{cell('BUILD', 'New-to-Credit Explorers', ms['Thin-file']['High'], ' · ⛔ hungry → PROTECT')}{cell('BUILD', 'New-to-Credit Explorers', ms['Thin-file']['Medium'], ' · ⛔')}{cell('BUILD', 'Dormant Unknowns', ms['Thin-file']['Low'], ' · ⛔')}</tr>
</table>""", unsafe_allow_html=True)
            st.markdown(f'<div class="rule" style="border-left-color:{RED}"><b>⛔ GUARD-RAIL 1</b> · Vulnerable <i>with</i> a stress signal '
                        f'is moved out of GROW into PROTECT (Stretched Seekers, {OFFICIAL["sub_segments"]["Stretched Seekers"]["customers"]:,}).</div>'
                        f'<div class="rule" style="border-left-color:{RED}"><b>⛔ GUARD-RAIL 2</b> · Thin-file with ≥ 2 rejections / 12m or '
                        f'≥ 3 applications / 90d is moved out of BUILD into PROTECT '
                        f'(Credit-Hungry Newcomers, {OFFICIAL["sub_segments"]["Credit-Hungry Newcomers"]["customers"]:,}).</div>',
                        unsafe_allow_html=True)
    with c[1]:
        with card():
            title('Opportunity map: risk vs response', 'bubble = customers · hover for the 95% CI of the bad rate')
            SS = OFFICIAL['sub_segments']
            POS = {'Prime Champions': 'top center', 'Loyal Core': 'middle right', 'Cautious Growers': 'bottom center',
                   'Sleeping Prime': 'middle left', 'Stretched Seekers': 'middle right', 'Credit-Hungry Stressed': 'top center',
                   'Silent Risk': 'middle right', 'Credit-Hungry Newcomers': 'middle right',
                   'New-to-Credit Explorers': 'middle left', 'Dormant Unknowns': 'middle right'}
            LBL = {'New-to-Credit Explorers': '<b>New-to-Credit Explorers</b>', 'Credit-Hungry Newcomers': 'CH Newcomers',
                   'Credit-Hungry Stressed': 'CH Stressed'}
            fig = go.Figure()
            for seg in SEG_ORDER:
                subs = [s for s in SUB_ORDER if SS[s]['segment'] == seg]
                fig.add_trace(go.Scatter(
                    x=[SS[s]['bt_newloan_bad'] for s in subs], y=[SS[s]['fwd_positive_per_100_calls'] for s in subs],
                    mode='markers+text', name=seg, text=[LBL.get(s_, s_) for s_ in subs],
                    textposition=[POS.get(s_, 'top center') for s_ in subs],
                    textfont=dict(size=9, color=INK, family='Inter, Segoe UI, sans-serif'),
                    marker=dict(size=[np.sqrt(SS[s]['customers']) / 16 + 7 for s in subs], color=SEG_COLOR[seg],
                                opacity=0.85, line=dict(color='#FFFFFF', width=2)),
                    customdata=[[SS[s]['customers'], OFFICIAL['sub_evidence'][s]['ci_low'], OFFICIAL['sub_evidence'][s]['ci_high']]
                                for s in subs],
                    hovertemplate='<b>%{text}</b><br>bad rate %{x:.1%} (95% CI %{customdata[1]:.1%}–%{customdata[2]:.1%})'
                                  '<br>%{y:.2f} positives / 100 calls<br>%{customdata[0]:,} customers<extra></extra>'))
            # --- "The Eager-Borrower Trap" called out explicitly ------------------------------------------
            nte = SS['New-to-Credit Explorers']
            fig.add_annotation(x=np.log10(nte['bt_newloan_bad']), y=nte['fwd_positive_per_100_calls'],
                               text='<b>⚠ THE EAGER-BORROWER TRAP</b><br>answers most: '
                                    f"{nte['fwd_positive_per_100_calls']:.2f} / 100 calls<br>but first loans go bad at "
                                    f"<b>{pct(nte['bt_newloan_bad'])}</b>",
                               showarrow=True, arrowhead=2, arrowcolor=RED, arrowwidth=1.6, ax=58, ay=-46,
                               font=dict(size=10, color='#A5101F'), align='left',
                               bgcolor='rgba(253,232,235,.95)', bordercolor=RED, borderwidth=1.4, borderpad=5)
            pcs = SS['Prime Champions']
            fig.add_annotation(x=np.log10(pcs['bt_newloan_bad']), y=pcs['fwd_positive_per_100_calls'],
                               text=f"<b>safe & responsive</b><br>bad {pct(pcs['bt_newloan_bad'])}",
                               showarrow=True, arrowhead=2, arrowcolor=GREEN, arrowwidth=1.4, ax=-10, ay=-46,
                               font=dict(size=9, color='#0B6B38'), align='left',
                               bgcolor='rgba(230,244,236,.95)', bordercolor=GREEN, borderwidth=1.2, borderpad=4)
            fig.update_xaxes(type='log', tickvals=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], tickformat='.1%',
                             range=[np.log10(0.0035), np.log10(0.28)],
                             title=dict(text='risk — bad rate on new loans (log scale)'))
            fig.update_yaxes(range=[0.0, 2.15], title=dict(text='response — positives / 100 calls'))
            show(apply_pro_theme(fig, 320, True, margin=dict(l=6, r=16, t=24, b=6)), 'nba_bubble')
    gap()
    c = st.columns([1, 0.85, 1.15], gap='medium')
    with c[0]:
        with card():
            title('Action mix of the selected customers', 'T1 → T2 → T3 override the segment action')
            nba_mix_chart(d, 'nba_mix', 265)
    with c[1]:
        with card():
            title('Channel rebalancing', 'telesales only for Active / Champion, max 5 attempts')
            cm = OFFICIAL['channel_mix']
            after = {'Telesales': sum(v for k_, v in cm.items() if k_.startswith('Telesales')),
                     'SMS / Zalo': sum(v for k_, v in cm.items() if k_.startswith('SMS')),
                     'In-app': sum(v for k_, v in cm.items() if k_.startswith('In-app')),
                     'Collections': sum(v for k_, v in cm.items() if k_.startswith('Collections'))}
            before = {'Telesales': 0.2551, 'SMS / Zalo': 0.4468, 'In-app': 0.2821, 'Collections': 0.016}
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(before), y=list(before.values()), name='Before audit', marker_color='#A8B4C4',
                                 text=[pct(v, 0) for v in before.values()], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{x}</b> before: %{y:.1%}<extra></extra>'))
            fig.add_trace(go.Bar(x=list(after), y=list(after.values()), name='Final engine', marker_color=NAVY,
                                 text=[pct(v, 0) for v in after.values()], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 hovertemplate='<b>%{x}</b> after: %{y:.1%}<extra></extra>'))
            fig.update_layout(barmode='group')
            fig.update_yaxes(showticklabels=False, range=[0, 0.72])
            show(apply_pro_theme(fig, 265, True), 'nba_chan')
    with c[2]:
        with card():
            title('Opportunity cost, both ways')
            vl = OFFICIAL['value_levers']
            cs = OFFICIAL['call_alloc_sub']
            table_html(pd.DataFrame([
                ['Stop sales calls to PROTECT',
                 f"{pct(vl['L3_sales_calls_to_protect_share'], 0)} of calls "
                 f"({OFFICIAL['call_alloc']['PROTECT']['pos_10k']:.0f} positives / 10k)",
                 f"~{vl['L3_bad_loans_avoided_per_1000_paused_offers_protect']:.0f} bad loans avoided / 1,000 paused offers; "
                 f"hours move to Prime ({cs['Prime Champions']['pos_10k']:.0f} / 10k)"],
                ['REACTIVATE & BUILD → digital', f"{pct(vl['L4_sales_calls_to_reactivate_build_share'], 0)} of calls",
                 f"Sleeping Prime answer only {OFFICIAL['call_alloc']['REACTIVATE']['pos_10k']:.0f} / 10k calls"],
                ['Structure Vulnerable instead of declining', f"risk {pct(vl['L7_bad_rate_cautious_growers'])} (Cautious Growers)",
                 'keeps about half of the loan volume in the FHS 40–60 band'],
                ['Tiny starter limits for thin-file', 'larger first tickets',
                 f"caps the loss where first loans go bad at {pct(vl['L8_bad_rate_new_to_credit_explorers'])}"]],
                columns=['Choice', 'What we give up', 'What we gain']))
    gap()
    # ---------------------------------------------------------------------------------------------------------
    # Campaign Target List Exporter + live audit of the engine's hard rules
    # ---------------------------------------------------------------------------------------------------------
    with card():
        title('📤 Campaign Target List Exporter', 'filter the scored book, audit the rules, download the campaign CSV')
        f = st.columns(5)
        def opts(c_):
            return sorted([x for x in d[c_].dropna().astype(str).unique() if x and x != 'nan'])
        fseg = f[0].multiselect('Segment', opts('SEGMENT'), key='ex_seg')
        fsub = f[1].multiselect('Sub-segment', opts('SUB_SEGMENT'), key='ex_sub')
        ftr = f[2].multiselect('Trigger', opts('NBA_TRIGGER'), key='ex_tr')
        fch = f[3].multiselect('Channel', opts('NBA_CHANNEL'), key='ex_ch')
        fpr = f[4].multiselect('Priority', opts('NBA_PRIORITY'), key='ex_pr')
        sel = d
        for col_, vals in [('SEGMENT', fseg), ('SUB_SEGMENT', fsub), ('NBA_TRIGGER', ftr), ('NBA_CHANNEL', fch), ('NBA_PRIORITY', fpr)]:
            if vals:
                sel = sel[sel[col_].astype(str).isin(vals)]
        m = st.columns(4)
        m[0].metric('Rows in the list', f'{len(sel):,}')
        m[1].metric('≈ Customers (full base)', f'{sel.WEIGHT.sum():,.0f}')
        tot_offer = float((sel.NBA_OFFER_VND.fillna(0) * sel.WEIGHT).sum())
        m[2].metric('≈ Offer amount', f'{tot_offer / 1e12:,.2f}T VND' if tot_offer >= 1e12 else f'{tot_offer / 1e9:,.1f}B VND')
        m[3].metric('Avg FHS · ES', f'{wmean(sel, "FHS"):.0f} · {wmean(sel, "ES"):.0f}' if len(sel) else '—')
        cols_ = ['SKP_CLIENT', 'SEGMENT', 'SUB_SEGMENT', 'FHS', 'FHS_TIER', 'ES', 'ES_LEVEL', 'NBA_TRIGGER', 'NBA_ACTION',
                 'NBA_PRODUCT', 'NBA_OFFER_VND', 'CREDIT_CEILING_VND', 'NBA_CHANNEL', 'NBA_TIMING', 'NBA_PRIORITY', 'EXPLANATION']
        cols_ = [c_ for c_ in cols_ if c_ in sel.columns]
        if len(sel):
            st.dataframe(sel.sort_values('NBA_PRIORITY_SCORE', ascending=False)[cols_].head(300), hide_index=True, **FULL, height=235)
        else:
            st.info('No customers match this combination — clear a filter above, or press 🔄 Reset All Filters.')
        dl = st.columns([1, 2.4])
        dl[0].download_button('⬇ Download Filtered Campaign CSV',
                              sel[cols_].to_csv(index=False).encode('utf-8-sig'),
                              file_name='ft_us_campaign_list.csv', mime='text/csv', key='nba_dl', **FULL,
                              disabled=not len(sel))
        # --- live audit: the three hard rules, re-checked on whatever is currently selected ---------------
        oc = OFFICIAL['output_checks']
        prot = sel[sel.SEGMENT.astype(str) == 'PROTECT']
        t3 = sel[sel.NBA_ACTION.astype(str).str.startswith('T3')]
        tel = sel[sel.NBA_CHANNEL.astype(str).str.startswith('Telesales')]
        a1 = bool((prot.CREDIT_CEILING_VND.fillna(0) <= 0).all()) if len(prot) else True
        a2 = bool(t3.FHS_TIER.astype(str).isin(['Resilient', 'Stable']).all()) if len(t3) else True
        a3 = bool(tel.ES_LEVEL.astype(str).isin(['Active', 'Champion']).all()) if len(tel) else True
        def mark(ok):
            return '✅' if ok else '⚠️'
        hours = ', '.join(f'{h}h' for h in OFFICIAL['best_call_hours'])
        dl[1].markdown(
            f'<div class="rule"><b>AUDIT RULES v6 (§8.2b &amp; §10b) — re-checked live on this list</b><br>'
            f'{mark(a1)} <b>100% of PROTECT customers carry CREDIT_CEILING_VND = 0</b> '
            f'({len(prot):,} PROTECT rows here · full-base violations: {oc["credit_ceiling_on_PROTECT"]})<br>'
            f'{mark(a2)} <b>T3 is locked to Resilient / Stable only</b> '
            f'({len(t3):,} T3 rows here · full-base violations: {oc["T3_outside_Resilient_Stable"]})<br>'
            f'{mark(a3)} <b>Telesales only calls Active / Champion</b>, max 5 attempts, golden hours {hours} '
            f'({len(tel):,} telesales rows here · full-base violations: {oc["telesales_to_REACTIVATE_BUILD_PROTECT"]})<br>'
            f'<span style="color:{INK_SOFT}">Full-base output checks: {oc["rows"]:,} rows · one row per customer · '
            f'0 nulls in required columns · every EXPLANATION names its own action.</span></div>',
            unsafe_allow_html=True)
    source(SRC)

# =============================================================================================================
# 05 · CUSTOMER 360°
# =============================================================================================================
elif page == PAGES[4]:
    header('05', 'Customer 360°', 'Every customer: one score pair, one action, one plain-English reason', slicers=False)
    PERSONAS = [('⭐ Prime Champion · T3 Maturity', dict(SUB_SEGMENT='Prime Champions', NBA_TRIGGER='T3 Loan maturing'), 95.0),
                ('🌱 Cautious Grower · Clean Vulnerable', dict(SUB_SEGMENT='Cautious Growers', NBA_TRIGGER='None'), 54.0),
                ('🛑 Credit-Hungry Stressed · PROTECT', dict(SUB_SEGMENT='Credit-Hungry Stressed', NBA_TRIGGER='None'), 38.0),
                ('🧭 New-to-Credit Explorer · BUILD', dict(SUB_SEGMENT='New-to-Credit Explorers', NBA_TRIGGER='None'), 40.0),
                ('🚨 Live Collection · T1', dict(NBA_TRIGGER='T1 Live collections'), 30.0)]

    def find_persona(cond, target):
        """Narrow progressively, but never to an empty frame: any condition that would empty the result is relaxed."""
        d_ = data
        for k_, v in cond.items():
            if k_ in d_.columns:
                m = d_[d_[k_].astype(str) == v]
                if len(m):
                    d_ = m
        if not len(d_):
            d_ = data
        try:
            idx = (d_.FHS.astype(float).fillna(60.0) - target).abs().argsort().values[0]
            return int(d_.iloc[idx].SKP_CLIENT)
        except Exception:
            return int(d_.iloc[0].SKP_CLIENT)

    def pick_persona(cond, target):
        """Widget callback → runs before the rerun, so writing the text_input's key is legal."""
        st.session_state['sel_id'] = str(find_persona(cond, target))

    if 'sel_id' not in st.session_state:
        st.session_state['sel_id'] = str(find_persona(PERSONAS[0][1], PERSONAS[0][2]))
    gap()
    cc = st.columns([1, 1, 1, 1, 1, 1.25])
    for i, (lab, cond, tf) in enumerate(PERSONAS):
        cc[i].button(lab, key=f'c360_p{i}', on_click=pick_persona, args=(cond, tf), **FULL,
                     help='Jump to a real customer of this type in the loaded file')
    sid = cc[5].text_input('SKP_CLIENT', key='sel_id', label_visibility='collapsed', placeholder='🔎 Type any SKP_CLIENT')
    try:
        rowc = data[data.SKP_CLIENT == int(str(sid).strip())]
    except (ValueError, TypeError):
        rowc = data.iloc[0:0]
    if rowc.empty:
        st.warning('This SKP_CLIENT is not in the loaded file. Press a persona button, or try one of: '
                   + ', '.join(map(str, data.SKP_CLIENT.head(5).tolist())))
    else:
        r_ = rowc.iloc[0]
        seg = str(r_.SEGMENT)
        prio = str(r_.NBA_PRIORITY)
        tier = str(r_.FHS_TIER)
        lvl = str(r_.ES_LEVEL)
        pcol = {'High': RED, 'Medium': AMBER, 'Low': SLATE}.get(prio, SLATE)
        st.markdown(f'<div class="hero">'
                    f'<span style="font-size:1.3rem;font-weight:800;margin-right:10px;">Customer {int(r_.SKP_CLIENT)}</span>'
                    f'<span class="tag" style="background:{SEG_COLOR.get(seg, SLATE)}">{seg}</span>'
                    f'<span class="tag" style="background:#334155">{r_.SUB_SEGMENT}</span>'
                    f'<span class="tag" style="background:#0E7490">Trigger · {r_.NBA_TRIGGER}</span>'
                    f'<span class="tag" style="background:{pcol}">Priority · {prio}</span></div>', unsafe_allow_html=True)

        def gauge(val, txt, bands, color):
            fig = go.Figure(go.Indicator(
                mode='gauge+number', value=float(val) if pd.notna(val) else 0,
                number=dict(font=dict(size=36, color=INK, family='Inter, Segoe UI, sans-serif')),
                title=dict(text=txt, font=dict(size=13, color=INK, family='Inter, Segoe UI, sans-serif')),
                gauge=dict(axis=dict(range=[0, 100], tickwidth=1, tickcolor=ZERO_C,
                                     tickfont=dict(size=9, color=INK_TICK, family='Inter, Segoe UI, sans-serif')),
                           bar=dict(color=color, thickness=0.3), bgcolor='rgba(0,0,0,0)',
                           steps=[dict(range=b_[:2], color=b_[2]) for b_ in bands], borderwidth=0)))
            return apply_gauge_theme(fig, 210)

        c = st.columns([1, 1, 1.2], gap='medium')
        with c[0]:
            with card():
                title('Financial health · C1')
                show(gauge(r_.FHS, f'FHS · {tier}',
                           [(0, 40, '#FDE2E5'), (40, 55, '#FDEBD3'), (55, 70, '#DCE7FD'), (70, 100, '#D7F0E2')],
                           TIER_COLOR.get(tier, SLATE)), 'c360_g1')
                table_html(pd.DataFrame([['Confidence', r_.FHS_CONFIDENCE],
                                         ['Stress flag', '🔴 stressed' if r_.STRESS_FLAG == 1 else '🟢 no'],
                                         ['Live collections', '🔴 yes' if r_.LIVE_STRESS_FLAG == 1 else '🟢 no'],
                                         ['➕ strongest positive', r_.FHS_POS_REASON],
                                         ['➖ strongest negative', r_.FHS_NEG_REASON]], columns=['Item', 'Value']))
        with c[1]:
            with card():
                title('Engagement · C2')
                cu = OFFICIAL['es_cuts']
                show(gauge(r_.ES, f'ES · {lvl}',
                           [(0, cu[0], '#E2E8F0'), (cu[0], cu[1], '#DBEAFE'), (cu[1], cu[2], '#BFDBFE'), (cu[2], 100, '#C7D2FE')],
                           LEVEL_COLOR.get(lvl, BLUE)), 'c360_g2')
                dimv = [r_.ES_RECENCY, r_.ES_FREQUENCY, r_.ES_INTERACTION, r_.ES_LEAD_ACTIVITY, r_.ES_PRODUCT_INTERACTION]
                if all(pd.notna(v) for v in dimv):
                    prf = OFFICIAL['es_profile']
                    dim_names = ['Recency', 'Frequency', 'Interaction', 'Lead', 'Product']
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=['R', 'F', 'I', 'L', 'P'], y=[prf['Warm'][k_] for k_ in 'RFILP'],
                                         name='Average Warm customer', marker_color='#A8B4C4', customdata=dim_names,
                                         hovertemplate='<b>%{customdata}</b> · average Warm: %{y:.1f}<extra></extra>'))
                    fig.add_trace(go.Bar(x=['R', 'F', 'I', 'L', 'P'], y=dimv, name='This customer',
                                         marker_color=LEVEL_COLOR.get(lvl, BLUE),
                                         text=[f'{v:.0f}' for v in dimv], textposition='outside', cliponaxis=False,
                                         textfont=dict(color=INK, size=10), customdata=dim_names,
                                         hovertemplate='<b>%{customdata}</b> · this customer: %{y:.1f}<extra></extra>'))
                    fig.update_layout(barmode='group')
                    fig.update_yaxes(range=[0, 112], showticklabels=False)
                    show(apply_pro_theme(fig, 175, True, margin=dict(l=6, r=12, t=22, b=6)), 'c360_rfilp')
                if pd.notna(r_.DAYS_SINCE_LAST_SELF_ACTIVITY):
                    source(f'Last self-initiated activity {r_.DAYS_SINCE_LAST_SELF_ACTIVITY:.0f} days ago')
        with c[2]:
            with card():
                title('Recommended Next Best Action · C3')
                st.markdown(f'<div style="color:{SEG_COLOR.get(seg, NAVY)};font-weight:800;font-size:1.08rem;margin:4px 0 8px 0;">'
                            f'{r_.NBA_ACTION}</div>', unsafe_allow_html=True)
                offer = (vnd(r_.NBA_OFFER_VND) if pd.notna(r_.NBA_OFFER_VND) and r_.NBA_OFFER_VND > 0
                         else '— no proactive credit / amount set at application')
                prio_txt = (f'{prio} (score {r_.NBA_PRIORITY_SCORE:.1f})' if pd.notna(r_.NBA_PRIORITY_SCORE) else prio)
                table_html(pd.DataFrame([['Product', r_.NBA_PRODUCT],
                                         ['Suggested offer', offer],
                                         ['Credit ceiling (engine)', vnd(r_.CREDIT_CEILING_VND)],
                                         ["Home Credit's own limit", vnd(r_.MAX_OFFER)],
                                         ['Channel', r_.NBA_CHANNEL],
                                         ['Timing', r_.NBA_TIMING],
                                         ['Priority', prio_txt],
                                         ['Detail', r_.NBA_DETAIL]], columns=['Field', 'Recommendation']))
                st.markdown('<p class="cs" style="text-align:left;margin-top:8px"><b>EXPLANATION</b> · verbatim from the output file</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="expl">{r_.EXPLANATION}</div>', unsafe_allow_html=True)
    source('Customer data: ' + data_src + ' · rules of notebook §8.2–8.2b · explanations are generated per customer')

# =============================================================================================================
# 06 · INSIGHTS & DECISIONS
# =============================================================================================================
else:
    header('06', 'Key Insights & Decisions', 'What the evidence says, what we ask the Board to decide, and how we prove it',
           slicers=False)
    i1, i2, i3, i4 = OFFICIAL['i1'], OFFICIAL['i2'], OFFICIAL['i3'], OFFICIAL['i4']
    vl = OFFICIAL['value_levers']
    gap()
    # ---------------------------------------------------------------------------------------------------------
    # The four signature insights
    # ---------------------------------------------------------------------------------------------------------
    st.markdown('<p class="ct" style="font-size:1.08rem;text-align:left">Four signature insights – what only C1 × C2 × C3 '
                'together reveal</p><p class="cs" style="text-align:left">Associations, not causes: every decision is '
                'launched against a random 10% hold-out</p>', unsafe_allow_html=True)
    SIG = [('I1 · Pull vs Push', f"{i1['PULL']['book_30d'] / i1['PUSH']['book_30d']:.1f}×",
            f"Customer-initiated (PULL) leads book <b>{pct(i1['PULL']['book_30d'])}</b> within 30 days, company-pushed leads only "
            f"<b>{pct(i1['PUSH']['book_30d'])}</b> – yet PUSH is {pct(i1['PUSH']['lead_share'], 0)} of all volume.",
            'Score the customer’s own initiative (ES), then spend the call budget where it is highest'),
           ('I2 · Credit hunger', f"{i2['4+']['bad_rate'] / i2['0']['bad_rate']:.1f}×",
            f"Shopping 4+ channels in 90 days lifts the bad rate from <b>{pct(i2['0']['bad_rate'], 2)}</b> to "
            f"<b>{pct(i2['4+']['bad_rate'], 2)}</b> and the rejection rate from {pct(i2['0']['reject_rate'], 0)} to "
            f"{pct(i2['4+']['reject_rate'], 0)}.",
            'Credit hunger is a guard-rail, not a buying signal → route to PROTECT'),
           ('I3 · Maturity window', f"{i3['matures in 0-30d']['rate'] / i3['matures in 91-180d']['rate']:.1f}×",
            f"A customer whose loan matures in 0–30 days applies again at <b>{pct(i3['matures in 0-30d']['rate'], 0)}</b>, "
            f"versus {pct(i3['matures in 91-180d']['rate'], 0)} at 91–180 days out.",
            f"Fire T3 re-loan offers in the 90-day window: {vl['L2_healthy_customers_maturing_90d']:,} healthy customers"),
           ('I4 · Ghost calling', f"{i4[1]['connect'] / i4[15]['connect']:.0f}×",
            f"Connect rate collapses from <b>{pct(i4[1]['connect'])}</b> on attempt 1 to <b>{pct(i4[15]['connect'])}</b> at "
            f"attempt 15+, while {i4[15]['attempts']:,} attempts are still spent there.",
            'Cap at 5 attempts, then switch to SMS / Zalo / in-app')]
    c = st.columns(4, gap='small')
    for i, (h, n_, t, r) in enumerate(SIG):
        with c[i]:
            st.markdown(f'<div class="big"><div class="hd">{h}</div><div class="n" style="font-size:2.2rem">{n_}</div>'
                        f'<div class="t">{t}</div>'
                        f'<div class="t" style="border-top:1px solid #E3E7EE;padding-top:8px;"><b>→ {r}</b></div></div>',
                        unsafe_allow_html=True)
    gap()
    c = st.columns([1, 1, 1, 1], gap='medium')
    with c[0]:
        with card():
            title('I1 · Who started it decides the outcome', 'books within 30 days, by lead initiator')
            order_i1 = ['PULL', 'WARM', 'PUSH']
            ICOL = {'PULL': GREEN, 'WARM': AMBER, 'PUSH': RED}
            fig = go.Figure(go.Bar(x=order_i1, y=[i1[s]['book_30d'] for s in order_i1],
                                   marker_color=[ICOL[s] for s in order_i1],
                                   text=[pct(i1[s]['book_30d']) for s in order_i1], textposition='outside', cliponaxis=False,
                                   textfont=dict(color=INK, size=11),
                                   customdata=[[i1[s]['leads'], i1[s]['lead_share']] for s in order_i1],
                                   hovertemplate='<b>%{x}</b><br>books %{y:.1%} in 30d<br>%{customdata[0]:,.0f} leads '
                                                 '(%{customdata[1]:.1%} of volume)<extra></extra>'))
            fig.update_yaxes(tickformat='.0%', range=[0, 0.52], title=dict(text='books within 30d'))
            show(apply_pro_theme(fig, 250, False), 'i6_i1')
    with c[1]:
        with card():
            title('I2 · Credit hunger raises risk', 'channels shopped in the last 90 days')
            ks = ['0', '1', '2', '3', '4+']
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ks, y=[i2[k_]['bad_rate'] for k_ in ks], name='bad rate',
                                 marker_color=[GREEN, '#86B98F', AMBER, '#E8762B', RED],
                                 text=[pct(i2[k_]['bad_rate'], 2) for k_ in ks], textposition='outside', cliponaxis=False,
                                 textfont=dict(color=INK, size=10),
                                 customdata=[[i2[k_]['applications'], i2[k_]['reject_rate']] for k_ in ks],
                                 hovertemplate='<b>%{x} channels</b><br>bad rate %{y:.2%}<br>%{customdata[0]:,.0f} applications · '
                                               'rejected %{customdata[1]:.1%}<extra></extra>'))
            fig.update_yaxes(tickformat='.1%', range=[0, 0.042], title=dict(text='bad rate on the loan taken'))
            fig.update_xaxes(title=dict(text='channels shopped / 90d'))
            show(apply_pro_theme(fig, 250, False), 'i6_i2')
    with c[2]:
        with card():
            title('I3 · The maturity window', 'share applying again, by where the loan clock stands')
            ks = ['matures in 0-30d', 'matures in 31-90d', 'matures in 91-180d', 'matures in 181-365d',
                  'matured ≤90d ago', 'matured 91-365d ago']
            lbl = ['0–30d', '31–90d', '91–180d', '181–365d', 'ended ≤90d', 'ended 91–365d']
            colr = [GREEN, GREEN, RED, RED, '#0E7490', '#0E7490']
            fig = go.Figure(go.Bar(x=lbl, y=[i3[k_]['rate'] for k_ in ks], marker_color=colr,
                                   text=[pct(i3[k_]['rate'], 0) for k_ in ks], textposition='outside', cliponaxis=False,
                                   textfont=dict(color=INK, size=10),
                                   customdata=[i3[k_]['n'] for k_ in ks],
                                   hovertemplate='<b>%{x}</b><br>applies again %{y:.1%}<br>%{customdata:,} customers<extra></extra>'))
            fig.update_yaxes(tickformat='.0%', range=[0, 0.78], title=dict(text='applies for a new loan'))
            show(apply_pro_theme(fig, 250, False), 'i6_i3')
    with c[3]:
        with card():
            title('I4 · Ghost calling', 'connect rate and attempts spent, by attempt number')
            atts = sorted(i4.keys())
            fig = go.Figure()
            fig.add_trace(go.Bar(x=atts, y=[i4[a]['attempts'] for a in atts], name='attempts spent',
                                 marker_color='#D8E0EA', yaxis='y2',
                                 hovertemplate='attempt %{x}<br>%{y:,} attempts<extra></extra>'))
            fig.add_trace(go.Scatter(x=atts, y=[i4[a]['connect'] for a in atts], name='connect rate', mode='lines+markers',
                                     line=dict(color=RED, width=2.5), marker=dict(size=6, color=RED),
                                     hovertemplate='attempt %{x}<br>connect %{y:.1%}<extra></extra>'))
            fig.add_vline(x=5.5, line=dict(color=NAVY, width=1.5, dash='dash'))
            fig.add_annotation(x=5.5, y=0.17, text='<b>cap at 5</b>', showarrow=False, xanchor='left',
                               font=dict(size=10, color=NAVY), bgcolor='rgba(255,255,255,.9)', borderpad=2)
            fig.update_yaxes(tickformat='.0%', title=dict(text='connect rate'))
            fig.update_xaxes(title=dict(text='attempt number on the same number'))
            apply_pro_theme(fig, 250, True)
            # the secondary axis is set last: apply_pro_theme must not re-show its ticks
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, showticklabels=False))
            show(fig, 'i6_i4')
    gap()
    # ---------------------------------------------------------------------------------------------------------
    # Opportunity cost both ways + the two A/B tests with their simulator
    # ---------------------------------------------------------------------------------------------------------
    c = st.columns([1.18, 1], gap='medium')
    with c[0]:
        with card():
            title('⚖️ Opportunity Cost — Both Ways', 'every decision is a trade, so both directions are priced')
            cs = OFFICIAL['call_alloc_sub']
            table_html(pd.DataFrame([
                ['Stop sales calls to PROTECT',
                 f"{pct(vl['L3_sales_calls_to_protect_share'], 0)} of all sales calls, "
                 f"{OFFICIAL['call_alloc']['PROTECT']['pos_10k']:.0f} positives / 10k",
                 f"~{vl['L3_bad_loans_avoided_per_1000_paused_offers_protect']:.0f} bad loans avoided per 1,000 paused offers",
                 f"Prime Champions return {cs['Prime Champions']['pos_10k']:.0f} positives / 10k for the same hour"],
                ['Keep calling PROTECT',
                 f"~{1000 * (vl['L3_bad_rate_protect'] - vl['L3_bad_rate_grow']):.0f} extra bad loans per 1,000 offers",
                 f"{pct(vl['L3_sales_calls_to_protect_share'], 0)} of the call budget stays where it is",
                 'Risk cost lands on the next 4 instalments, not on this month'],
                ['Move REACTIVATE & BUILD to digital',
                 f"{pct(vl['L4_sales_calls_to_reactivate_build_share'], 0)} of calls",
                 f"Sleeping Prime answer only {OFFICIAL['call_alloc']['REACTIVATE']['pos_10k']:.0f} / 10k calls",
                 'SMS / Zalo / in-app costs a fraction of an agent hour'],
                ['Structure the FHS 40–60 band instead of declining',
                 f"risk {pct(vl['L7_bad_rate_cautious_growers'])} on {vl['L7_cautious_growers_structured']:,} Cautious Growers",
                 'keeps about half of the loan volume in that band',
                 'Smaller ticket, ≤ 12 months, down payment ≥ 30%'],
                ['Tiny starter limits for thin-file',
                 'larger first tickets on eager newcomers',
                 f"caps the loss where first loans go bad at {pct(vl['L8_bad_rate_new_to_credit_explorers'])}",
                 f"{vl['L8_credit_hungry_newcomers']:,} Credit-Hungry Newcomers verified before lending"]],
                columns=['Decision', 'What we give up', 'What we gain', 'Why it is worth it']))
    with c[1]:
        with card():
            title('🧪 Prove it first: the two A/B tests', 'random 10% hold-out on each · move the sliders to size the prize')
            ta, tb = st.columns(2)
            with ta:
                st.markdown(f'<div style="color:{INK};font-weight:700;font-size:.80rem">TEST A · T3 MATURITY RE-LOAN</div>',
                            unsafe_allow_html=True)
                upl = st.slider('Incremental booking uplift (pp)', 0.5, 6.0, 2.5, 0.5, key='ab_t3',
                                help='Extra percentage points of booking in the treated group versus the hold-out')
            with tb:
                st.markdown(f'<div style="color:{INK};font-weight:700;font-size:.80rem">TEST B · PROTECT PAUSE</div>',
                            unsafe_allow_html=True)
                cov = st.slider('Share of PROTECT offers paused', 0, 100, 100, 10, key='ab_protect',
                                help='How much of the Credit-Hungry Stressed pool actually gets its credit push paused')
            HOLD = 0.10                                     # the 10% hold-out never receives the treatment
            pop_a = vl['L2_healthy_customers_maturing_90d']
            treat_a = pop_a * (1 - HOLD)
            extra_loans = treat_a * upl / 100
            pop_b = vl['L3_credit_hungry_customers']
            treat_b = pop_b * (1 - HOLD) * cov / 100
            risk_gap = vl['L3_bad_rate_credit_hungry'] - vl['L3_bad_rate_grow']
            bad_blocked = treat_b * risk_gap
            m = st.columns(2)
            stat(m[0], 'Extra loans booked (Test A)', f'{extra_loans:,.0f}', GREEN)
            stat(m[1], 'Bad loans blocked (Test B)', f'{bad_blocked:,.0f}', RED)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=['Test A · T3 re-loan', 'Test B · PROTECT pause'],
                                 x=[treat_a, treat_b], orientation='h', name='treated (90%)',
                                 marker_color=[GREEN, RED],
                                 text=[f'{treat_a / 1e3:,.0f}k', f'{treat_b / 1e3:,.0f}k'], textposition='inside',
                                 insidetextfont=dict(color='#FFFFFF', size=10),
                                 hovertemplate='<b>%{y}</b><br>%{x:,.0f} customers treated<extra></extra>'))
            fig.add_trace(go.Bar(y=['Test A · T3 re-loan', 'Test B · PROTECT pause'],
                                 x=[pop_a * HOLD, pop_b * HOLD], orientation='h', name='hold-out (10%)',
                                 marker_color='#A8B4C4',
                                 hovertemplate='<b>%{y}</b><br>%{x:,.0f} customers held out<extra></extra>'))
            fig.update_layout(barmode='stack')
            fig.update_xaxes(showticklabels=False, showgrid=False)
            fig.update_yaxes(tickfont=dict(color=INK, size=11, family='Inter, Segoe UI, sans-serif'))
            show(apply_pro_theme(fig, 150, True, margin=dict(l=6, r=12, t=22, b=6)), 'i6_ab')
            table_html(pd.DataFrame([
                ['Test A population', f"{pop_a:,} healthy customers whose loan ends ≤ 90 days"],
                ['Test B population', f"{pop_b:,} Credit-Hungry Stressed ({pct(vl['L3_bad_rate_credit_hungry'])} bad)"],
                ['Commercial KPI', 'positives per 100 contacts, measured at day 60'],
                ['Risk KPI', 'bad rate on the next 4 instalments'],
                ['Scale-up rule', 'beat the hold-out on both KPIs, and be profitable after losses']],
                columns=['Design', 'Value']))
    gap()
    c = st.columns([1.1, 1, 0.9], gap='medium')
    with c[0]:
        with card():
            title('Risk × engagement: where each customer belongs')
            fig = go.Figure()
            quad = [(0.5, 1.5, 'GROW', 'healthy & listening<br>pre-approved offers'),
                    (1.5, 1.5, 'REACTIVATE', 'healthy & silent<br>digital win-back'),
                    (0.5, 0.5, 'PROTECT', 'fragile<br>no credit push'),
                    (1.5, 0.5, 'BUILD', 'no history<br>starter limits')]
            fig.update_xaxes(visible=False, range=[0, 2])
            fig.update_yaxes(visible=False, range=[0, 2])
            apply_pro_theme(fig, 280, False)
            for x, y, s, t in quad:
                fig.add_shape(type='rect', x0=x - 0.48, x1=x + 0.48, y0=y - 0.46, y1=y + 0.46,
                              fillcolor=SEG_COLOR[s], line=dict(width=0), opacity=0.94)
                fig.add_annotation(x=x, y=y + 0.12, text=f'<b>{s}</b> · {pct(OFFICIAL["segments"][s]["share"])}',
                                   showarrow=False, font=dict(color='#FFFFFF', size=14, family='Inter, Segoe UI, sans-serif'))
                fig.add_annotation(x=x, y=y - 0.15, text=t, showarrow=False,
                                   font=dict(color='#FFFFFF', size=11, family='Inter, Segoe UI, sans-serif'))
            show(fig, 'i6_quad')
    with c[1]:
        with card():
            title('Three decisions for the Board')
            dec = [('01', 'Adopt a 3-band credit appetite',
                    'Grow ≥ 60 (48% of loans, 0.71% bad) · structure 40–60 · protect &lt; 40 (avoid 10.6% of bad loans for 3.4% '
                    'of volume). Owner: CRO.'),
                   ('02', 'Re-route contact effort',
                    'No sales calls to PROTECT; REACTIVATE &amp; BUILD digital; telesales only Active / Champion, max 5 attempts '
                    '→ ~821k calls freed. Owner: Sales.'),
                   ('03', 'Run the engine monthly',
                    'Re-score all 1,000,528 customers; one action each, tracked on this dashboard. Owner: BI / CVM.')]
            for n_, h, t in dec:
                st.markdown(f'<div style="display:flex;gap:12px;margin:6px 0 10px 0;">'
                            f'<div style="background:{NAVY};color:#FFFFFF !important;border-radius:10px;min-width:44px;height:44px;'
                            f'display:flex;align-items:center;justify-content:center;font-weight:800;">{n_}</div>'
                            f'<div style="color:{INK_BODY};font-size:.84rem;line-height:1.4;"><b style="color:{INK}">{h}</b><br>{t}</div></div>',
                            unsafe_allow_html=True)
    with c[2]:
        with card():
            title('S4 · Attempt-cap menu', 'pick a cap: attempts saved vs positive outcomes lost')
            CAPS = {3: (0.5289, 0.2467, 1.1764), 5: (0.4171, 0.1955, 1.0155), 8: (0.2849, 0.1440, 0.8807), 10: (0.2188, 0.1179, 0.8308)}
            n_cap = st.select_slider('Stop calling a number after N unanswered attempts', options=[3, 5, 8, 10], value=5, key='i6_cap')
            a_, l_, p_ = CAPS[n_cap]
            fig = go.Figure(go.Bar(x=['Attempts saved', 'Positives lost'], y=[a_, l_], marker_color=[BLUE, RED],
                                   text=[pct(a_, 0), pct(l_, 0)], textposition='outside', cliponaxis=False,
                                   textfont=dict(color=INK, size=11),
                                   hovertemplate='<b>%{x}</b>: %{y:.1%}<extra></extra>'))
            fig.update_yaxes(tickformat='.0%', range=[0, 0.64])
            show(apply_pro_theme(fig, 195, False), 'i6_cap_fig')
            st.markdown(f'<p class="cs" style="text-align:left">Positives per 100 attempts: <b>0.74 → {p_:.2f}</b>. '
                        f'The cap only pays if the freed hours are re-used on GROW and T3 lists.</p>', unsafe_allow_html=True)
    gap()
    c = st.columns(5, gap='small')
    evid = [('0.661', 'FHS out-of-time AUC', GREEN), ('7.9×', 'risk spread across tiers', NAVY),
            ('91,102', 'financially stressed', RED), ('177,252', 'healthy, loan maturing ≤ 90d', BLUE),
            ('9.7%', 'telesales share after rebalancing', AMBER)]
    for i, (n_, t, col) in enumerate(evid):
        with c[i]:
            kpi('Evidence', n_, t, col, col)
    gap()
    takeaway([('Associations, not causes',
               'Every lift tells us where to test; each action launches against a 10% hold-out (Important Note 4).'),
              ('Governance',
               'Recalibrate FHS levels quarterly (out-of-time level 2.8× training); fix CALL timestamps; start recording '
               'confirmed fraud – stacking is a hypothesis, not a finding.')],
             head='⚖️ Read with care')
    source(SRC)
