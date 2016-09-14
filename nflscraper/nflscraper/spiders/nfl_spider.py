# Run this spider in the command line with the following command: scrapy crawl pfr -o command_results.csv

# To use the pipeline simply uncomment the pipeline lines in the settings.py file and just run scrapy crawl pfr, but no headers will populate





import scrapy

from scrapy import Selector
from nflscraper.items import NflscraperItem
from datetime import date

class NFLScraperSpider(scrapy.Spider):
	name = "pfr"
	allowed_domains = ['pro-football-reference.com']
	def start_requests(self):
		yield scrapy.Request("http://www.pro-football-reference.com/years/2015/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2015/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2014/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2013/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2012/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2011/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2010/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2009/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2008/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2007/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2006/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2005/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2004/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2003/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2002/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2001/games.htm", self.parse)
		yield scrapy.Request("http://www.pro-football-reference.com/years/2000/games.htm", self.parse)

	def parse(self,response):
		for href in response.xpath('//a[contains(text(),"boxscore")]/@href'):
			item = NflscraperItem()
			url = response.urljoin(href.extract())
			request = scrapy.Request(url, callback=self.parse_dir_contents)
			request.meta['item'] = item
			yield request

	def parse_dir_contents(self,response):
		item = response.meta['item']
		# Code to pull out JS comment - http://stackoverflow.com/questions/38781357/pro-football-reference-team-stats-xpath/38781659#38781659
		team_stats_text = response.xpath('//div[@id="all_team_stats"]//comment()').extract()[0]
		team_stats_selector = Selector(text=team_stats_text[4:-3].strip())
		
		# Pull out JS comment for Vegas info
		game_info_text = response.xpath('//div[@id="all_game_info"]//comment()').extract()[0]
		game_info_selector = Selector(text=game_info_text[4:-3].strip())
		# Item population
		# Game Date
		DateDict = {"Jan" : 1, "Feb" : 2, "Aug" : 8, "Sep" : 9, "Oct" : 10, "Nov" : 11, "Dec" : 12}
		gamedate = response.xpath('//*[@id="content"]/div[2]/div[3]/div[1]/text()').extract()[0]
		gamedate = gamedate[gamedate.find(" ")+1:]
		month = DateDict[gamedate[:gamedate.find(" ")]]
		day_year = gamedate[gamedate.find(" ")+1:]
		day = int(day_year[:day_year.find(",")])
		year = int(gamedate[-4:])
		item['game_date'] = date(year,month,day)
		
		# Team names
		item['home_team'] = response.xpath('//*[@id="content"]/table/tbody/tr[2]/td[2]/a/text()').extract()[0]
		item['away_team'] = response.xpath('//*[@id="content"]/table/tbody/tr[1]/td[2]/a/text()').extract()[0]
		
		# Scores
		item['home_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[2]/td[last()]/text()').extract()[0]
		item['away_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[1]/td[last()]/text()').extract()[0]
		
		# Vegas info
		item['vegasline'] = game_info_selector.xpath('//*[@id="game_info"]//td/text()').extract()[-2]
		item['overunder'] = game_info_selector.xpath('//*[@id="game_info"]//td/text()').extract()[-1]
		
		# Offensive total yards
		item['home_oyds'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[2]/text()').extract()[0]
		item['away_oyds'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[1]/text()').extract()[0]
		
		# Rush yards extracted from total rushing values
		hrush = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[2]/td[2]/text()').extract()[0]
		hrush = hrush[hrush.find("-")+1:]
		item['home_rush'] = hrush[:hrush.find("-")]
		arush = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[2]/td[1]/text()').extract()[0]
		arush = arush[arush.find("-")+1:]
		item['away_rush'] = arush[:arush.find("-")]

		# Rush attempts extracted from total rushing values
		hrush = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[2]/td[2]/text()').extract()[0]
		item['hrush_att'] = hrush[:hrush.find("-")]
		arush = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[2]/td[1]/text()').extract()[0]
		item['arush_att'] = arush[:arush.find("-")]
		
		# Total sacks and sack yards
		hsack = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[4]/td[2]/text()').extract()[0]
		item['hsack_yds'] = hsack[hsack.find("-")+1:]
		asack = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[4]/td[1]/text()').extract()[0]
		item['asack_yds'] = asack[asack.find("-")+1:]
		item['home_sack'] = hsack[:hsack.find("-")]
		item['away_sack'] = asack[:asack.find("-")]
		
		# Net passing yards (total passing yards minus yards sacked)
		item['home_pass'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[5]/td[2]/text()').extract()[0]
		item['away_pass'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[5]/td[1]/text()').extract()[0]

		# Total pass stats
		item['hpass_tot'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[3]/td[2]/text()').extract()[0]
		item['apass_tot'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[3]/td[1]/text()').extract()[0]
		
		# Turnovers
		item['home_turn'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[2]/text()').extract()[0]
		item['away_turn'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[1]/text()').extract()[0]
		
		# Number of penalties and penalty yards
		hpens = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[9]/td[2]/text()').extract()[0]
		item['home_pens'] = hpens[:hpens.find("-")]
		apens = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[9]/td[1]/text()').extract()[0]
		item['away_pens'] = apens[:apens.find("-")]
		item['hpens_yds'] = hpens[hpens.find("-")+1:]
		item['apens_yds'] = apens[apens.find("-")+1:]
		
		# Third down conversion rate
		hthird = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[10]/td[2]/text()').extract()[0]
		hthirdconv = float(hthird[:hthird.find("-")])
		hthirdatt = float(hthird[hthird.find("-")+1:])
		item['home_third'] = hthirdconv / hthirdatt
		athird = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[10]/td[1]/text()').extract()[0]
		athirdconv = float(athird[:athird.find("-")])
		athirdatt = float(athird[athird.find("-")+1:])
		item['away_third'] = athirdconv / athirdatt
		
		# Fourth down conversion rate
		hfour = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[10]/td[2]/text()').extract()[0]
		hfourconv = float(hfour[:hfour.find("-")])
		hfouratt = float(hfour[hfour.find("-")+1:])
		item['home_four'] = hfourconv / hfouratt
		afour = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[10]/td[1]/text()').extract()[0]
		afourconv = float(afour[:afour.find("-")])
		afouratt = float(afour[afour.find("-")+1:])
		item['away_four'] = afourconv / afouratt
		
		# Time of possession
		item['home_poss'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[12]/td[2]/text()').extract()[0]
		item['away_poss'] = team_stats_selector.xpath('//*[@id="team_stats"]/tbody/tr[12]/td[1]/text()').extract()[0]
		yield item