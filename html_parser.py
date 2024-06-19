import re
from bs4 import BeautifulSoup

FEATURE_PATTERN = r'(\d+\.\d+%?|\d+%?|[^\d.%]+)'

def get_statistics_dict(html):
    data_dict = {}
    soup = BeautifulSoup(html, 'lxml')
    # print(f'match title: {soup.title.string}')
    scores = soup.select('div.detailScore__wrapper')[0].get_text().split('-')
    data_dict['result'] = 'WIN' if scores[0] > scores[1] else 'LOSS' if scores[0] < scores[1] else 'DRAW'
    data_dict[f'score_home'] = scores[0]
    data_dict[f'score_away'] = scores[1]
    
    rows = soup.select('div.section div._row_bn1w5_8')
    for element in rows:
        text = element.get_text()
        regex_matches = re.findall(FEATURE_PATTERN, text)
        # != 대신 is not 쓰면 안됨... 값이 같더라도 메모리 위치가 달라서 걸림
        if text != ''.join(regex_matches):
            raise Exception(f'regex match failure: {text} | {regex_matches}')

        data_dict[f'{regex_matches[1]}_home'] = regex_matches[0]
        data_dict[f'{regex_matches[1]}_away'] = regex_matches[2]

    return data_dict

def get_match_summary_url_list(html):
    soup = BeautifulSoup(html, 'lxml')
    elements = soup.select('div.sportName.soccer [class*="event__match event__match--withRowLink event__match--static"] a')      
    return [a['href'] for a in elements]
