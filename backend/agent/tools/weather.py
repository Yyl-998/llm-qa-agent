"""天气查询工具：基于 Open-Meteo 免费接口，无需 API Key。"""
import httpx
from langchain_core.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WMO_CODES = {
    0: "晴", 1: "基本晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气和未来 3 天预报。输入应为城市中文名，如：北京、上海、杭州。"""
    try:
        geo = httpx.get(
            GEOCODE_URL,
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"未找到城市「{city}」，请确认城市名称。"
        loc = results[0]

        resp = httpx.get(
            FORECAST_URL,
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        cur = data["current"]
        lines = [
            f"{loc.get('name', city)}（{loc.get('country', '')}{loc.get('admin1', '')}）当前天气：",
            f"- 温度 {cur['temperature_2m']}°C（体感 {cur['apparent_temperature']}°C），{WMO_CODES.get(cur['weather_code'], '未知')}",
            f"- 湿度 {cur['relative_humidity_2m']}%，风速 {cur['wind_speed_10m']} km/h",
            "",
            "未来 3 天预报：",
        ]
        daily = data["daily"]
        for i, d in enumerate(daily["time"]):
            desc = WMO_CODES.get(daily["weather_code"][i], "未知")
            lines.append(
                f"- {d}：{desc}，{daily['temperature_2m_min'][i]}~{daily['temperature_2m_max'][i]}°C"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败：{e}"
