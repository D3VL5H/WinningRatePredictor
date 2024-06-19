
import time
import re
import csv
import utils
import html_parser as htmlp
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

LEAGUE_LIST = ['germany/bundesliga', 'spain/laliga', 'italy/serie-a',
               'england/premier-league', 'france/ligue-1']

BASE_URL = "https://www.livesport.com/kr/soccer/"
season_range = range(2019, 2024)
#season_range = range(2014, 2019)
season_list = [f'{year}-{year + 1}' for year in season_range]

url_list = [f'{BASE_URL}{league}-{season}/results/' for league in LEAGUE_LIST for season in season_list]

# K리그는 유럽 리그랑 시즌 이름이 달라서 따로 지정해야함
# url_list = [f'https://www.livesport.com/kr/soccer/south-korea/k-league-1-{season}/results/' for season in range(2018, 2025)]

# js 렌더링이 필요해서 requests->selenium으로 변경
opt = Options()
# chrome창 안띄움, gpu가속 비활성화
opt.add_argument("--headless=new")
opt.add_argument("--disable-gpu")
# ssl 오류 무시
opt.add_argument('--ignore-certificate-errors')
opt.add_argument('--ignore-ssl-errors')
srv = Service(ChromeDriverManager().install())
drv = webdriver.Chrome(service=srv, options=opt)

# 페이지로 이동 후 로드될때까지 대기
def nav_to_url(url, is_wait = True):
    drv.get(url)
    if is_wait == False:
        return
    WebDriverWait(drv, 10).until(
        lambda d: drv.execute_script('return document.readyState') == 'complete'
    )

# main
for url in url_list:
    print(url)
    nav_to_url(url)
    while True:
        try:
            # '더 많은 경기 보기' 버튼 찾기 (클릭이 가능해질때까지 최대 10초간 대기)
            load_more_btn = WebDriverWait(drv, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.event__more.event__more--static'))
            )
            # 버튼 위치까지 스크롤 내리고 클릭
            # scrollIntoView 사용 시 광고에 가려질때가 있음
            # drv.execute_script(f'arguments[0].scrollIntoView(true);', load_more_btn)
            drv.execute_script("""
                var element = arguments[0];
                var elementRect = element.getBoundingClientRect();
                var absoluteElementTop = elementRect.top + window.pageYOffset;
                var middle = absoluteElementTop - (window.innerHeight / 2);
                window.scrollTo(0, middle);
                """, load_more_btn)

            load_more_btn.click()
            print("load_more_btn clicked")

            # 더 이상 버튼이 없다면(모든 경기결과가 표시되었다면) break
        except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as ex:
            print(f"no more buttons: {ex.msg}")
            break
        except ElementClickInterceptedException as ex:
            print(f"button click failure: {ex.msg}")
            # blocking_element = driver.find_element(By.CSS_SELECTOR, ".boxOverContent__bannerLink")
            # drv.execute_script("arguments[0].remove();", blocking_element)
            WebDriverWait(drv, 1)
            continue
    
    file_name = url.replace(BASE_URL, '').replace('/', '_')

    match_list = htmlp.get_match_summary_url_list(drv.page_source)
    utils.create_dir_if_not_exists('match_url_backup')
    with open(f'match_url_backup/{file_name}.txt', 'w') as f:
        f.writelines(f'{m}\n' for m in match_list)

    data_dict_list = []
    features = set()
    # progressbar 띄움
    with tqdm(total=len(match_list), desc=file_name, unit='Match', dynamic_ncols=True, leave=True) as pbar:
        for m in match_list:
            nav_to_url(f'{m}/match-statistics/')
            try:
                WebDriverWait(drv, 10).until(            
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'div.section div._row_bn1w5_8'))
                )
            except:
                # 통계 자체가 존재하지 않는 경기의 경우 발생할 수 있음
                tqdm.write('timeout')
                continue
            
            # print대신 tqdm.write를 사용해야 progressbar가 안밀림
            tqdm.write(f'match title: {drv.title}')
            data_dict = htmlp.get_statistics_dict(drv.page_source)
            data_dict_list.append(data_dict)
            features.update(data_dict.keys())            
            # 서버 부하 방지
            time.sleep(0.2)
            pbar.update(1)

    utils.create_dir_if_not_exists('data')    
    with open(f'data/{file_name}.csv', 'w', newline='') as f:        
        writer = csv.DictWriter(f, fieldnames=sorted(features))
        writer.writeheader()
        writer.writerows(data_dict_list)
        print(f'{f.name} created')   


drv.quit()