# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class NflscraperItem(scrapy.Item):
	home_score = scrapy.Field()
	away_score = scrapy.Field()
	home_oyds = scrapy.Field()
	away_oyds = scrapy.Field()
	home_dyds = scrapy.Field()
	away_dyds = scrapy.Field()
	home_turn = scrapy.Field()
	away_turn = scrapy.Field()