"""Customer-facing Chinese names for every Olist product category.

One table for the whole product: the order list, the planner's view of an order, confirmation
cards and replies all use it. The four demo categories keep their existing names. Olist has two
near-duplicate codes (home_appliances_2, home_comfort_2); customers see the same name as for the
original code. "unknown" covers products without a category in the source data.
"""

CATEGORY_LABELS: dict[str, str] = {
    "agro_industry_and_commerce": "农业与工商用品",
    "air_conditioning": "空调设备",
    "art": "艺术品",
    "arts_and_craftmanship": "手工艺品",
    "audio": "音响耳机",
    "auto": "汽车用品",
    "baby": "母婴用品",
    "bed_bath_table": "家居用品",
    "books_general_interest": "大众图书",
    "books_imported": "进口图书",
    "books_technical": "专业图书",
    "cds_dvds_musicals": "CD 与音乐影碟",
    "christmas_supplies": "圣诞用品",
    "cine_photo": "摄影摄像器材",
    "computers": "电脑整机",
    "computers_accessories": "电脑配件",
    "consoles_games": "游戏机与游戏",
    "construction_tools_construction": "建筑工具",
    "construction_tools_lights": "照明建材",
    "construction_tools_safety": "安全防护用品",
    "cool_stuff": "创意好物",
    "costruction_tools_garden": "园林工具",
    "costruction_tools_tools": "五金工具",
    "diapers_and_hygiene": "纸尿裤与卫生用品",
    "drinks": "饮品",
    "dvds_blu_ray": "DVD 与蓝光影碟",
    "electronics": "电子产品",
    "fashio_female_clothing": "女装",
    "fashion_bags_accessories": "包袋与配饰",
    "fashion_childrens_clothes": "童装",
    "fashion_male_clothing": "男装",
    "fashion_shoes": "鞋靴",
    "fashion_sport": "运动服饰",
    "fashion_underwear_beach": "内衣与沙滩装",
    "fixed_telephony": "座机电话",
    "flowers": "鲜花绿植",
    "food": "食品",
    "food_drink": "食品饮料",
    "furniture_bedroom": "卧室家具",
    "furniture_decor": "家具与家居装饰",
    "furniture_living_room": "客厅家具",
    "furniture_mattress_and_upholstery": "床垫与布艺家具",
    "garden_tools": "园艺工具",
    "health_beauty": "健康与美容用品",
    "home_appliances": "家用电器",
    "home_appliances_2": "家用电器",
    "home_comfort_2": "居家舒适用品",
    "home_confort": "居家舒适用品",
    "home_construction": "家装建材",
    "housewares": "家庭日用品",
    "industry_commerce_and_business": "工商业用品",
    "kitchen_dining_laundry_garden_furniture": "厨房餐厅洗衣与庭院家具",
    "la_cuisine": "厨房用品",
    "luggage_accessories": "箱包旅行用品",
    "market_place": "综合商品",
    "music": "音乐用品",
    "musical_instruments": "乐器",
    "office_furniture": "办公家具",
    "party_supplies": "派对用品",
    "pc_gamer": "游戏电脑",
    "perfumery": "香水香氛",
    "pet_shop": "宠物用品",
    "portateis_cozinha_e_preparadores_de_alimentos": "便携厨电与料理机",
    "security_and_services": "安防服务",
    "signaling_and_security": "标识与安全用品",
    "small_appliances": "小家电",
    "small_appliances_home_oven_and_coffee": "烤箱与咖啡机",
    "sports_leisure": "运动休闲用品",
    "stationery": "文具",
    "tablets_printing_image": "平板与打印设备",
    "telephony": "手机通讯",
    "toys": "玩具",
    "watches_gifts": "手表与礼品",
    "unknown": "其他商品",
}


def category_label(category: str | None) -> str:
    """The name a customer sees; a multi-category order ("a、b") gets each part named."""
    text = (category or "").strip()
    if not text:
        return "商品"
    parts = [p.strip() for p in text.split("、") if p.strip()]
    return "、".join(CATEGORY_LABELS.get(p, p.replace("_", " ")) for p in parts)
