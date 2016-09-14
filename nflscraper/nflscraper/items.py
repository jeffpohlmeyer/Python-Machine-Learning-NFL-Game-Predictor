# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class NflscraperItem(scrapy.Item):
	game_date = scrapy.Field()
	home_team = scrapy.Field()
	away_team = scrapy.Field()
	home_score = scrapy.Field()
	away_score = scrapy.Field()
	vegasline = scrapy.Field()
	overunder = scrapy.Field()
	home_rush = scrapy.Field()
	away_rush = scrapy.Field()
	hrush_att = scrapy.Field()
	arush_att = scrapy.Field()
	home_sack = scrapy.Field()
	away_sack = scrapy.Field()
	hsack_yds = scrapy.Field()
	asack_yds = scrapy.Field()
	home_pass = scrapy.Field()
	away_pass = scrapy.Field()
	hpass_tot = scrapy.Field()
	apass_tot = scrapy.Field()
	home_oyds = scrapy.Field()
	away_oyds = scrapy.Field()
	home_turn = scrapy.Field()
	away_turn = scrapy.Field()
	home_pens = scrapy.Field()
	away_pens = scrapy.Field()
	hpens_yds = scrapy.Field()
	apens_yds = scrapy.Field()
	home_third = scrapy.Field()
	away_third = scrapy.Field()
	home_four = scrapy.Field()
	away_four = scrapy.Field()
	home_poss = scrapy.Field()
	away_poss = scrapy.Field()