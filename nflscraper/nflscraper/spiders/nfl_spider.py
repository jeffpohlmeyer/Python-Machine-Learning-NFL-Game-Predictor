import scrapy

from scrapy import Selector
from nflscraper.items import NflscraperItem

class NFLScraperSpider(scrapy.Spider):
	name = "pfr"
	allowed_domains = ['www.pro-football-reference.com/']
	start_urls = [
		#"http://www.pro-football-reference.com/years/2015/games.htm"
		"http://www.pro-football-reference.com/boxscores/201510110tam.htm"
	]

	def parse(self,response):
		item = NflscraperItem()
		extracted_text = response.xpath('//div[@id="all_team_stats"]//comment()').extract()[0]
		new_selector = Selector(text=extracted_text[4:-3].strip())
		item['home_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[2]/td[last()]/text()').extract()[0].strip()
		item['away_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[1]/td[last()]/text()').extract()[0].strip()
		item['home_oyds'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[2]/text()').extract()[0].strip()
		item['away_oyds'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[1]/text()').extract()[0].strip()
		item['home_dyds'] = item['away_oyds']
		item['away_dyds'] = item['home_oyds']
		item['home_turn'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[2]/text()').extract()[0].strip()
		item['away_turn'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[1]/text()').extract()[0].strip()
		yield item
"""		for href in response.xpath('//a[contains(text(),"boxscore")]/@href'):
			url = response.urljoin(href.extract())
			yield scrapy.Request(url, callback=self.parse_dir_contents)

	def parse_dir_contents(self,response):
		item = NflscraperItem()
		extracted_text = response.xpath('//div[@id="all_team_stats"]//comment()').extract()[0]
		new_selector = Selector(text=extracted_text[4:-3].strip())
		item['home_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[2]/td[last()]/text()').extract()[0].strip()
		item['away_score'] = response.xpath('//*[@id="content"]/table/tbody/tr[1]/td[last()]/text()').extract()[0].strip()
		item['home_oyds'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[2]/text()').extract()[0].strip()
		item['away_oyds'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[6]/td[1]/text()').extract()[0].strip()
		item['home_dyds'] = item['away_oyds']
		item['away_dyds'] = item['home_oyds']
		item['home_turn'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[2]/text()').extract()[0].strip()
		item['away_turn'] = new_selector.xpath('//*[@id="team_stats"]/tbody/tr[8]/td[1]/text()').extract()[0].strip()
		yield item
"""