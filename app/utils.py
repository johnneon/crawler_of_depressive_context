import json
import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Загружает результаты анализа из JSON-файла
    
    Args:
        file_path: путь к файлу с результатами
        
    Returns:
        List[Dict[str, Any]]: список профилей с данными
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Ошибка при загрузке результатов: {e}")
        return []

def create_dataframe(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Создает DataFrame из списка профилей для удобного анализа
    
    Args:
        profiles: список профилей с данными
        
    Returns:
        pd.DataFrame: датафрейм с данными профилей
    """
    rows = []
    
    for profile in profiles:
        # Базовая информация о пользователе
        user_data = {
            "user_id": profile.get("user_id", ""),
            "name": profile.get("name", ""),
            "screen_name": profile.get("screen_name", ""),
            "post_count": len(profile.get("posts", [])),
            "sex": profile.get("sex", "Не указан"),
            "city": profile.get("city", "Не указан"),
            "followers_count": profile.get("followers_count", 0),
            "friends_count": profile.get("friends_count", 0),
            "alcohol": profile.get("alcohol", "Не указан"),
            "smoking": profile.get("smoking", "Не указан"),
            "life_main": profile.get("life_main", "Не указан"),
            "people_main": profile.get("people_main", "Не указан"),
        }
        
        # Если есть предсказание депрессии
        if "label" in profile:
            user_data["depression_label"] = profile["label"]
            user_data["depression_probability"] = profile.get("probability", 0)
        
        rows.append(user_data)
    
    return pd.DataFrame(rows)

def plot_depression_distribution(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Создает график распределения предсказаний депрессии
    
    Args:
        df: датафрейм с данными профилей
        
    Returns:
        Optional[plt.Figure]: объект графика или None в случае ошибки
    """
    if "depression_label" not in df.columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Распределение классов
        sns.countplot(data=df, x="depression_label", ax=ax)
        ax.set_title("Распределение предсказаний депрессии")
        ax.set_xlabel("Класс (0 - нет депрессии, 1 - есть депрессии)")
        ax.set_ylabel("Количество профилей")
        
        # Добавляем проценты
        total = len(df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2.,
                    height + 0.1,
                    f'{height/total*100:.1f}%',
                    ha="center") 
        
        return fig
    except Exception as e:
        st.warning(f"Не удалось создать график распределения: {e}")
        return None

def plot_probability_histogram(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Создает гистограмму вероятностей депрессии
    
    Args:
        df: датафрейм с данными профилей
        
    Returns:
        Optional[plt.Figure]: объект графика или None в случае ошибки
    """
    if "depression_probability" not in df.columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Гистограмма вероятностей
        sns.histplot(data=df, x="depression_probability", bins=20, ax=ax)
        ax.set_title("Распределение вероятностей депрессии")
        ax.set_xlabel("Вероятность депрессии")
        ax.set_ylabel("Количество профилей")
        
        # Добавляем вертикальную линию на уровне 0.5
        ax.axvline(x=0.5, color='r', linestyle='--', label='Порог классификации (0.5)')
        ax.legend()
        
        return fig
    except Exception as e:
        st.warning(f"Не удалось создать гистограмму вероятностей: {e}")
        return None

def display_profile_card(profile: Dict[str, Any]):
    """
    Отображает карточку профиля пользователя
    
    Args:
        profile: словарь с данными профиля
    """
    # Создаем рамку для карточки профиля
    with st.container(border=True):
        # Основная информация
        col1, col2 = st.columns([2, 1])
        
        with col1:
            name = profile.get("name", "Нет имени")
            screen_name = profile.get("screen_name", "")
            
            st.subheader(f"{name} ({screen_name})")
            
            # Базовая информация
            st.write(f"**Город:** {profile.get('city', 'Не указан')}")
            st.write(f"**Пол:** {profile.get('sex', 'Не указан')}")
            st.write(f"**Подписчики:** {profile.get('followers_count', 0)}")
            st.write(f"**Друзья:** {profile.get('friends_count', 0)}")
        
        with col2:
            # Если есть предсказание депрессии
            if "label" in profile:
                label = profile["label"]
                probability = profile.get("probability", 0)
                
                # Цветной индикатор
                if label == 1:
                    st.error("**Признаки депрессии обнаружены**")
                    st.metric("Вероятность депрессии", f"{probability:.2%}")
                else:
                    st.success("**Признаки депрессии не обнаружены**")
                    st.metric("Вероятность депрессии", f"{probability:.2%}")
        
        # Дополнительная информация
        with st.expander("Дополнительная информация"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Отношение к алкоголю:** {profile.get('alcohol', 'Не указан')}")
                st.write(f"**Отношение к курению:** {profile.get('smoking', 'Не указан')}")
            with col2:
                st.write(f"**Главное в жизни:** {profile.get('life_main', 'Не указан')}")
                st.write(f"**Главное в людях:** {profile.get('people_main', 'Не указан')}")
        
        # Посты пользователя
        if "posts" in profile and profile["posts"]:
            with st.expander(f"Посты пользователя ({len(profile['posts'])})"):
                for i, post in enumerate(profile["posts"]):
                    st.markdown(f"**Пост {i+1}** - {post.get('date', 'Нет даты')}")
                    st.markdown(post.get("text", "Нет текста"))
                    st.markdown("---") 