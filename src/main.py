import json
import random
import copy
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# Import the files module for Colab
from google.colab import files

# ==============================================================================
# PARÁMETROS DEL ALGORITMO GENÉTICO (Existing parameters)
# ==============================================================================
POBLACION_SIZE = 300
GENERACIONES = 300
TASA_CRUCE = 0.65
TASA_MUTACION = 0.01
ELITISMO_COUNT = 15
TOURNAMENT_SIZE = 15
TASA_MUTACION_ESCRITORIO_REL = 0.75
TASA_MUTACION_DIA_REL = 0.20
TASA_MUTACION_ELIMINACION_REL = 0.05
TASA_MUTACION_ADICION = 0.10


# ==============================================================================
# CARGA Y PREPROCESAMIENTO DE DATOS (Existing functions)
# ==============================================================================

def cargar_datos(nombre_archivo: str):
    """Carga datos desde un archivo JSON y maneja errores."""
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            return json.load(archivo)
    except FileNotFoundError:
        print(f"Error Crítico: El archivo de datos '{nombre_archivo}' no fue encontrado.")
        return None
    except json.JSONDecodeError:
        print(f"Error Crítico: El archivo '{nombre_archivo}' contiene un JSON inválido.")
        return None
    except Exception as e:
        print(f"Error Crítico: Ocurrió un error inesperado al cargar el archivo: {e}")
        return None

# ==============================================================================
# FUNCIONES DEL ALGORITMO GENÉTICO (Existing functions)
# ==============================================================================

def crear_individuo(employees, days, dias_por_empleado, escritorios_por_empleado):
    """Crea un individuo (solución) inicial, priorizando restricciones duras."""
    asignacion = {}
    escritorios_ocupados = set()
    lista_empleados = list(employees)
    random.shuffle(lista_empleados)

    for e in lista_empleados:
        dias_preferidos = dias_por_empleado.get(e, [])
        random.shuffle(dias_preferidos)
        for d in dias_preferidos:
            escritorios_validos = escritorios_por_empleado.get(e, [])
            opciones = [esc for esc in escritorios_validos if (d, esc) not in escritorios_ocupados]
            if opciones:
                escritorio_elegido = random.choice(opciones)
                asignacion[(e, d)] = escritorio_elegido
                escritorios_ocupados.add((d, escritorio_elegido))

    for e in lista_empleados:
        if not any(key[0] == e for key in asignacion):
            dias_asignados_actualmente = dias_por_empleado.get(e, [])
            dias_no_preferidos = [d for d in days if d not in dias_asignados_actualmente]
            random.shuffle(dias_no_preferidos)
            for d_alt in dias_no_preferidos:
                escritorios_validos = escritorios_por_empleado.get(e, [])
                opciones = [esc for esc in escritorios_validos if (d_alt, esc) not in escritorios_ocupados]
                if opciones:
                    escritorio_elegido = random.choice(opciones)
                    asignacion[(e, d_alt)] = escritorio_elegido
                    escritorios_ocupados.add((d_alt, escritorio_elegido))
                    break
    return asignacion


def calcular_fitness(asignacion, datos_completos):
    """Calcula la aptitud de un individuo (menor penalización es mejor)."""
    penalizacion = 0

    employees = datos_completos['employees']
    empleados_por_grupo = datos_completos['empleados_por_grupo']
    dias_por_empleado = datos_completos['dias_por_empleado']
    escritorios_por_empleado = datos_completos['escritorios_por_empleado']
    grupo_por_empleado = datos_completos['grupo_por_empleado']
    zona_por_escritorio = datos_completos['zona_por_escritorio']

    ocupacion = defaultdict(list)
    zonas_dia = defaultdict(lambda: defaultdict(list))
    dias_asignados = defaultdict(set)

    for (e, d), desk in asignacion.items():
        ocupacion[(d, desk)].append(e)
        zona = zona_por_escritorio.get(desk)
        if zona:
            zonas_dia[d][zona].append(e)
        dias_asignados[e].add(d)

    for empleados_conflicto in ocupacion.values():
        if len(empleados_conflicto) > 1: penalizacion += 10000 * (len(empleados_conflicto) - 1)
    for (e, d), desk in asignacion.items():
        if desk not in escritorios_por_empleado.get(e, []): penalizacion += 10000
    for e in employees:
        if e not in dias_asignados: penalizacion += 5000
    for grupo, miembros in empleados_por_grupo.items():
        dias_reunion = defaultdict(int)
        for miembro in miembros:
            for dia in dias_asignados.get(miembro, []): dias_reunion[dia] += 1
        max_reunidos = max(dias_reunion.values()) if dias_reunion else 0
        if max_reunidos < len(miembros): penalizacion += 10000 * (len(miembros) - max_reunidos)
    for d, zonas in zonas_dia.items():
        for z, empleados_en_zona in zonas.items():
            grupos_presentes = defaultdict(int)
            for e in empleados_en_zona:
                grupo_e = grupo_por_empleado.get(e)
                if grupo_e: grupos_presentes[grupo_e] += 1
            for count in grupos_presentes.values():
                if count == 1: penalizacion += 500
    for e in employees:
        dias_preferidos_e = set(dias_por_empleado.get(e, []))
        dias_asignados_e = dias_asignados.get(e, set())
        dias_no_cubiertos = dias_preferidos_e - dias_asignados_e
        penalizacion += 200 * len(dias_no_cubiertos)
        dias_no_preferidos_asignados = dias_asignados_e - dias_preferidos_e
        penalizacion += 100 * len(dias_no_preferidos_asignados)
    for e in employees:
        escritorios_usados = {desk for (emp, _), desk in asignacion.items() if emp == e}
        if len(escritorios_usados) > 1: penalizacion += 50 * (len(escritorios_usados) - 1)
    return penalizacion

def reparar_individuo(individuo, escritorios_por_empleado):
    """Repara conflictos de asignación en un individuo."""
    ocupacion = defaultdict(list)
    for (e, d), desk in individuo.items():
        ocupacion[(d, desk)].append(e)
    for (d, desk), empleados_conflicto in ocupacion.items():
        if len(empleados_conflicto) > 1:
            for e_reasignar in empleados_conflicto[1:]:
                del individuo[(e_reasignar, d)]
                escritorios_ocupados_dia = {v for (k_e, k_d), v in individuo.items() if k_d == d}
                opciones = [esc for esc in escritorios_por_empleado.get(e_reasignar, []) if esc not in escritorios_ocupados_dia]
                if opciones:
                    individuo[(e_reasignar, d)] = random.choice(opciones)
    return individuo

def cruzar(padre1, padre2, escritorios_por_empleado):
    """Realiza cruce uniforme y repara al hijo resultante."""
    hijo = {}
    claves_padre1 = set(padre1.keys())
    claves_padre2 = set(padre2.keys())
    for clave in claves_padre1:
        if random.random() < 0.5: hijo[clave] = padre1[clave]
    for clave in claves_padre2:
        if random.random() < 0.5: hijo[clave] = padre2[clave]
    return reparar_individuo(hijo, escritorios_por_empleado)

def mutar(individuo, employees, days, escritorios_por_empleado):
    """Aplica mutaciones a un individuo para introducir diversidad."""
    mutado = copy.deepcopy(individuo)
    if not mutado: return reparar_individuo(mutado, escritorios_por_empleado)
    claves_a_mutar = list(mutado.keys())
    for (e, d) in claves_a_mutar:
        if (e, d) not in mutado: continue
        if random.random() < TASA_MUTACION:
            tipo_mutacion = random.random()
            if tipo_mutacion < TASA_MUTACION_ESCRITORIO_REL:
                escritorios_ocupados_dia = {v for (k_e, k_d), v in mutado.items() if k_d == d and k_e != e}
                opciones = [esc for esc in escritorios_por_empleado.get(e, []) if esc not in escritorios_ocupados_dia]
                if opciones: mutado[(e, d)] = random.choice(opciones)
            elif tipo_mutacion < TASA_MUTACION_ESCRITORIO_REL + TASA_MUTACION_DIA_REL:
                escritorio_actual = mutado.pop((e, d))
                dias_posibles = [day for day in days if day != d]
                random.shuffle(dias_posibles)
                reasignado = False
                for nuevo_d in dias_posibles:
                    escritorios_ocupados_nuevo_dia = {v for (k_e, k_d), v in mutado.items() if k_d == nuevo_d}
                    if escritorio_actual not in escritorios_ocupados_nuevo_dia:
                        mutado[(e, nuevo_d)] = escritorio_actual
                        reasignado = True
                        break
                if not reasignado: mutado[(e, d)] = escritorio_actual
            else:
                del mutado[(e, d)]
    if random.random() < TASA_MUTACION_ADICION:
        e_candidato = random.choice(list(employees))
        dias_asignados_e = {k_d for (k_e, k_d) in mutado.keys() if k_e == e_candidato}
        dias_disponibles_e = [day for day in days if day not in dias_asignados_e]
        if dias_disponibles_e:
            d_nuevo = random.choice(dias_disponibles_e)
            escritorios_ocupados_dia = {v for (k_e, k_d), v in mutado.items() if k_d == d_nuevo}
            opciones = [esc for esc in escritorios_por_empleado.get(e_candidato, []) if esc not in escritorios_ocupados_dia]
            if opciones: mutado[(e_candidato, d_nuevo)] = random.choice(opciones)
    return reparar_individuo(mutado, escritorios_por_empleado)

def seleccionar_padre_por_torneo(poblacion_con_puntuaciones):
    """Selecciona un padre mediante el método de torneo."""
    participantes = random.sample(poblacion_con_puntuaciones, TOURNAMENT_SIZE)
    ganador = min(participantes, key=lambda x: x[1])
    return ganador[0]

# ==============================================================================
# FUNCIONES DE REPORTE Y ANÁLISIS (Modified and New)
# ==============================================================================

def reportar_resultados(mejor_asignacion, datos_completos):
    """
    Calcula los KPIs de la mejor solución encontrada y los retorna.
    """
    employees = datos_completos['employees']
    desks = datos_completos['desks']
    days = datos_completos['days']
    escritorios_por_empleado = datos_completos['escritorios_por_empleado']
    dias_por_empleado = datos_completos['dias_por_empleado']
    empleados_por_grupo = datos_completos['empleados_por_grupo']
    grupo_por_empleado = datos_completos['grupo_por_empleado']
    zona_por_escritorio = datos_completos['zona_por_escritorio']

    ocupacion = defaultdict(list)
    zonas_dia = defaultdict(lambda: defaultdict(list))
    dias_asignados = defaultdict(set)
    escritorios_ocupados_por_dia = defaultdict(set)

    for (e, d), desk in mejor_asignacion.items():
        ocupacion[(d, desk)].append(e)
        zona = zona_por_escritorio.get(desk)
        if zona:
            zonas_dia[d][zona].append(e)
        dias_asignados[e].add(d)
        escritorios_ocupados_por_dia[d].add(desk)

    kpis = {}

    # --- KPIs de Restricciones Duras ---
    total_empleados = len(employees)
    empleados_asignados = len(dias_asignados)
    kpis['empleados_con_asignacion_pct'] = (empleados_asignados / total_empleados) * 100 if total_empleados > 0 else 0

    conflictos = sum(1 for v in ocupacion.values() if len(v) > 1)
    kpis['escritorios_con_sobre_asignacion'] = conflictos

    asignaciones_permitidas = sum(1 for (e, day_assigned), desk_assigned in mejor_asignacion.items() if desk_assigned in escritorios_por_empleado.get(e, []))
    kpis['asignaciones_en_escritorios_permitidos_pct'] = (asignaciones_permitidas / len(mejor_asignacion)) * 100 if len(mejor_asignacion) > 0 else 0

    # --- KPIs de Restricciones Suaves ---
    cumplimiento_dias_pref = sum(len(dias_asignados.get(e, set()) & set(dias_por_empleado.get(e, []))) / len(dias_por_empleado.get(e, [])) for e in employees if dias_por_empleado.get(e))
    kpis['cumplimiento_promedio_dias_preferidos_pct'] = (cumplimiento_dias_pref / len(employees)) * 100 if len(employees) > 0 else 0

    dias_no_pref = sum(len(dias_asignados.get(e, set()) - set(dias_por_empleado.get(e, []))) for e in employees)
    kpis['promedio_dias_asignados_no_preferidos'] = dias_no_pref / total_empleados if total_empleados > 0 else 0

    escritorio_unico = sum(1 for e in employees if len({desk for (emp, _), desk in mejor_asignacion.items() if emp == e}) == 1)
    kpis['empleados_con_escritorio_unico_pct'] = (escritorio_unico / total_empleados) * 100 if total_empleados > 0 else 0

    # --- KPIs de Colaboración ---
    total_asignaciones = len(mejor_asignacion)
    instancias_aislamiento = 0
    for d_zonas in zonas_dia.values():
        for z_empleados in d_zonas.values():
            grupos_presentes = defaultdict(int)
            for e in z_empleados:
                if grupo_por_empleado.get(e): grupos_presentes[grupo_por_empleado.get(e)] += 1
            instancias_aislamiento += sum(count for count in grupos_presentes.values() if count == 1)

    kpis['porcentaje_asignaciones_aisladas'] = (instancias_aislamiento / total_asignaciones) * 100 if total_asignaciones > 0 else 0

    kpis['detalle_reuniones_grupo'] = []
    grupos_con_reunion_completa = 0
    for grupo, miembros in empleados_por_grupo.items():
        if not miembros:
            kpis['detalle_reuniones_grupo'].append(f"Grupo '{grupo}': No tiene miembros asignados.")
            continue

        dias_reunion = defaultdict(int)
        for m in miembros:
            for d in dias_asignados.get(m, []):
                dias_reunion[d] += 1

        mejor_dia_reunion = None
        max_miembros_presentes = 0

        if dias_reunion:
            mejor_dia_reunion = max(dias_reunion, key=dias_reunion.get)
            max_miembros_presentes = dias_reunion[mejor_dia_reunion]

        if max_miembros_presentes == len(miembros):
            grupos_con_reunion_completa += 1
            kpis['detalle_reuniones_grupo'].append(f"Grupo '{grupo}': ¡Reunión completa! ({len(miembros)}/{len(miembros)} miembros) el día '{mejor_dia_reunion}'.")
        elif mejor_dia_reunion:
            porcentaje_asistencia = (max_miembros_presentes / len(miembros)) * 100
            kpis['detalle_reuniones_grupo'].append(f"Grupo '{grupo}': {max_miembros_presentes}/{len(miembros)} miembros ({porcentaje_asistencia:.2f}%) el día '{mejor_dia_reunion}'.")
        else:
            kpis['detalle_reuniones_grupo'].append(f"Grupo '{grupo}': Ningún miembro se reunió en un día común.")

    kpis['grupos_con_reunion_completa_pct'] = (grupos_con_reunion_completa / len(empleados_por_grupo)) * 100 if empleados_por_grupo else 0

    # --- KPIs de Eficiencia ---
    total_desks = len(desks)
    total_days = len(days)

    kpis['capacidad_utilizada_por_dia'] = {}
    total_ocupados_acumulado = 0
    if total_desks > 0 and total_days > 0:
        for d in days:
            occupied_desks_today = len(escritorios_ocupados_por_dia[d])
            total_ocupados_acumulado += occupied_desks_today
            daily_capacity_percentage = (occupied_desks_today / total_desks) * 100
            kpis['capacidad_utilizada_por_dia'][d] = {
                'ocupados': occupied_desks_today,
                'total_escritorios': total_desks,
                'porcentaje': daily_capacity_percentage
            }

        kpis['porcentaje_capacidad_utilizada_global'] = (total_ocupados_acumulado / (total_desks * total_days)) * 100
    else:
        kpis['porcentaje_capacidad_utilizada_global'] = 0

    return kpis


def generar_reporte_json(mejor_asignacion, datos_completos, nombre_archivo="reporte_asignaciones.json"):
    """
    Genera un archivo JSON con la solución detallada.
    """
    reporte = {
        "asignaciones_por_dia": defaultdict(list),
        "dias_preferidos_por_empleado": {e: datos_completos['dias_por_empleado'].get(e, []) for e in datos_completos['employees']}
    }
    for (e, d), desk in mejor_asignacion.items():
        reporte["asignaciones_por_dia"][d].append({"empleado": e, "escritorio": desk})

    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)
    print(f"\nReporte JSON detallado generado en: '{nombre_archivo}'")


def generar_reporte_excel(mejor_asignacion, kpis, nombre_archivo="reporte_asignaciones.xlsx"):
    """
    Genera un archivo Excel con la mejor asignación y los KPIs.
    """
    # Hoja de Asignaciones
    data_asignaciones = []
    for (empleado, dia), escritorio in mejor_asignacion.items():
        data_asignaciones.append({'Empleado': empleado, 'Día': dia, 'Escritorio': escritorio})
    df_asignaciones = pd.DataFrame(data_asignaciones)

    # Hoja de KPIs
    data_kpis = [
        ["KPI", "Valor", "Unidad", "Comentario"],
        ["Empleados con asignación", f"{kpis['empleados_con_asignacion_pct']:.2f}", "%", "Ideal: 100%"],
        ["Escritorios con sobre-asignación", kpis['escritorios_con_sobre_asignacion'], "", "Ideal: 0"],
        ["Asignaciones en escritorios permitidos", f"{kpis['asignaciones_en_escritorios_permitidos_pct']:.2f}", "%", "Ideal: 100%"],
        ["Cumplimiento promedio de días preferidos", f"{kpis['cumplimiento_promedio_dias_preferidos_pct']:.2f}", "%", "Más alto es mejor"],
        ["Promedio de días asignados no-preferidos", f"{kpis['promedio_dias_asignados_no_preferidos']:.2f}", "", "Más bajo es mejor"],
        ["Empleados con escritorio único", f"{kpis['empleados_con_escritorio_unico_pct']:.2f}", "%", "Más alto es mejor"],
        ["Porcentaje de asignaciones aisladas", f"{kpis['porcentaje_asignaciones_aisladas']:.2f}", "%", "Más bajo es mejor"],
        ["Grupos con reunión completa", f"{kpis['grupos_con_reunion_completa_pct']:.2f}", "%", "Más alto es mejor"],
        ["Capacidad Utilizada GLOBAL", f"{kpis['porcentaje_capacidad_utilizada_global']:.2f}", "%", "Más alto es mejor"]
    ]

    # Agregar detalle de capacidad por día
    for day, info in kpis['capacidad_utilizada_por_dia'].items():
        data_kpis.append([f"Capacidad Día '{day}'", f"{info['ocupados']}/{info['total_escritorios']} ({info['porcentaje']:.2f})", "%", ""])


    df_kpis = pd.DataFrame(data_kpis[1:], columns=data_kpis[0])

    with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
        df_asignaciones.to_excel(writer, sheet_name='Asignaciones', index=False)
        df_kpis.to_excel(writer, sheet_name='KPIs', index=False)

    print(f"Reporte Excel generado en: '{nombre_archivo}'")


def generar_graficos(kpis, output_dir="."):
    """
    Genera gráficos a partir de los KPIs y los guarda como imágenes.
    Retorna una lista de rutas a los archivos de imagen generados.
    """
    graficos_generados = []

    # Gráfico de Capacidad Utilizada por Día
    days = list(kpis['capacidad_utilizada_por_dia'].keys())
    percentages = [kpis['capacidad_utilizada_por_dia'][d]['porcentaje'] for d in days]

    if days and percentages:
        plt.figure(figsize=(10, 6))
        plt.bar(days, percentages, color='skyblue')
        plt.xlabel('Día')
        plt.ylabel('Porcentaje de Capacidad Utilizada (%)')
        plt.title('Capacidad Utilizada de Escritorios por Día')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--')
        img_path = f"{output_dir}/capacidad_utilizada_por_dia.png"
        plt.savefig(img_path)
        plt.close()
        graficos_generados.append(img_path)

    # Gráfico de KPIs de Calidad (ejemplo: cumplimiento días preferidos, escritorio único)
    kpi_nombres = ["Cumplimiento Días Preferidos", "Empleados con Escritorio Único", "Asignaciones Aisladas"]
    kpi_valores = [
        kpis['cumplimiento_promedio_dias_preferidos_pct'],
        kpis['empleados_con_escritorio_unico_pct'],
        kpis['porcentaje_asignaciones_aisladas']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(kpi_nombres, kpi_valores, color=['lightgreen', 'lightcoral', 'lightsalmon'])
    plt.xlabel('KPI')
    plt.ylabel('Porcentaje (%)')
    plt.title('KPIs Clave de Calidad y Colaboración')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

    img_path = f"{output_dir}/kpis_calidad_colaboracion.png"
    plt.savefig(img_path)
    plt.close()
    graficos_generados.append(img_path)

    print(f"Gráficos generados en: '{output_dir}'")
    return graficos_generados


def generar_reporte_pdf(mejor_penalizacion, mejor_asignacion, kpis, graficos_paths, nombre_archivo="reporte_final.pdf"):
    """
    Genera un reporte PDF completo con resultados, KPIs y gráficos.
    """
    doc = SimpleDocTemplate(nombre_archivo, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Título
    story.append(Paragraph("<b>Reporte de Asignación de Escritorios Optimizada</b>", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Mejor Penalización Encontrada
    story.append(Paragraph(f"<b>Mejor Puntuación de Penalización Encontrada:</b> {mejor_penalizacion}", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    # Mejor Asignación Encontrada
    story.append(Paragraph("<b>--- Mejor Asignación Encontrada ---</b>", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    asignacion_data = [["Empleado", "Día", "Escritorio"]]
    for (e, d), desk in mejor_asignacion.items():
        asignacion_data.append([e, d, desk])

    # Limitar la tabla de asignaciones para PDF si es muy larga
   # if len(asignacion_data) > 50: # Adjust this limit as needed
    #    story.append(Paragraph("<i>Mostrando las primeras 50 asignaciones por brevedad. Ver reporte Excel para la lista completa.</i>", styles['Normal']))
     #   asignacion_data = asignacion_data[:51] # Header + 50 rows

    table = Table(asignacion_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))


    # KPIs
    story.append(Paragraph("<b>==================================================</b>", styles['h2']))
    story.append(Paragraph("<b>KPIs DE LA MEJOR SOLUCIÓN ENCONTRADA</b>", styles['h2']))
    story.append(Paragraph("<b>==================================================</b>", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    # KPIs de Cumplimiento
    story.append(Paragraph("<b>--- 1. KPIs de Cumplimiento (Restricciones Duras) ---</b>", styles['h3']))
    story.append(Paragraph(f"<b>• Empleados con asignación:</b> {kpis['empleados_con_asignacion_pct']:.2f}% (Ideal: 100%)", styles['Normal']))
    story.append(Paragraph(f"<b>• Escritorios con sobre-asignación:</b> {kpis['escritorios_con_sobre_asignacion']} (Ideal: 0)", styles['Normal']))
    story.append(Paragraph(f"<b>• Asignaciones en escritorios permitidos:</b> {kpis['asignaciones_en_escritorios_permitidos_pct']:.2f}% (Ideal: 100%)", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # KPIs de Calidad
    story.append(Paragraph("<b>--- 2. KPIs de Calidad (Restricciones Suaves) ---</b>", styles['h3']))
    story.append(Paragraph(f"<b>• Cumplimiento promedio de días preferidos:</b> {kpis['cumplimiento_promedio_dias_preferidos_pct']:.2f}% (Más alto es mejor)", styles['Normal']))
    story.append(Paragraph(f"<b>• Promedio de días asignados no-preferidos:</b> {kpis['promedio_dias_asignados_no_preferidos']:.2f} (Más bajo es mejor)", styles['Normal']))
    story.append(Paragraph(f"<b>• Empleados con escritorio único:</b> {kpis['empleados_con_escritorio_unico_pct']:.2f}% (Más alto es mejor)", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # KPIs de Colaboración
    story.append(Paragraph("<b>--- 3. KPIs de Colaboración ---</b>", styles['h3']))
    story.append(Paragraph(f"<b>• Porcentaje de asignaciones aisladas:</b> {kpis['porcentaje_asignaciones_aisladas']:.2f}% (Más bajo es mejor)", styles['Normal']))
    story.append(Paragraph("<b> Detalle de Reuniones de Grupo:</b>", styles['Normal']))
    for item in kpis['detalle_reuniones_grupo']:
        story.append(Paragraph(f"    - {item}", styles['Normal']))
    story.append(Paragraph(f"<b>• Grupos con reunión completa:</b> {kpis['grupos_con_reunion_completa_pct']:.2f}% (Más alto es mejor)", styles['Normal']))
    story.append(Spacer(1, 0.1 * inch))

    # KPIs de Eficiencia
    story.append(Paragraph("<b>--- 4. KPIs de Eficiencia ---</b>", styles['h3']))
    story.append(Paragraph("<b> Capacidad Utilizada por Día:</b>", styles['Normal']))
    for day, info in kpis['capacidad_utilizada_por_dia'].items():
        story.append(Paragraph(f"    - Día '{day}': {info['ocupados']}/{info['total_escritorios']} escritorios ({info['porcentaje']:.2f}%)", styles['Normal']))
    story.append(Paragraph(f"<b>• Porcentaje de Capacidad Utilizada GLOBAL:</b> {kpis['porcentaje_capacidad_utilizada_global']:.2f}% (Más alto es mejor)", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Gráficos
    story.append(Paragraph("<b>--- Gráficos ---</b>", styles['h2']))
    story.append(Spacer(1, 0.2 * inch))

    for img_path in graficos_paths:
        try:
            img = Image(img_path, width=5.5*inch, height=3.5*inch) # Adjust size as needed
            story.append(img)
            story.append(Spacer(1, 0.1 * inch))
        except FileNotFoundError:
            story.append(Paragraph(f"<i>Error: No se pudo cargar el gráfico {img_path}.</i>", styles['Normal']))

    doc.build(story)
    print(f"Reporte PDF generado en: '{nombre_archivo}'")


# ==============================================================================
# BUCLE PRINCIPAL DEL ALGORITMO (Modified to call new report functions and print KPIs)
# ==============================================================================

def ejecutar_algoritmo_genetico(datos):
    """
    Orquesta la ejecución completa del algoritmo genético.
    """
    # Desempaquetar datos y añadir estructuras preprocesadas
    datos['employees'] = datos.get("Employees", [])
    datos['desks'] = datos.get("Desks", [])
    datos['days'] = datos.get("Days", [])
    datos['groups'] = datos.get("Groups", [])
    datos['zones'] = datos.get("Zones", [])
    datos['desks_por_zona'] = datos.get("Desks_Z", {})
    datos['escritorios_por_empleado'] = datos.get("Desks_E", {})
    datos['empleados_por_grupo'] = datos.get("Employees_G", {})
    datos['dias_por_empleado'] = datos.get("Days_E", {})
    datos['zona_por_escritorio'] = {d: z for z, dl in datos['desks_por_zona'].items() for d in dl}
    datos['grupo_por_empleado'] = {e: g for g, ml in datos['empleados_por_grupo'].items() for e in ml}

    # 1. Inicialización de la Población
    print("Generando población inicial...")
    poblacion = [crear_individuo(datos['employees'], datos['days'], datos['dias_por_empleado'], datos['escritorios_por_empleado']) for _ in range(POBLACION_SIZE)]

    mejor_asignacion_global = None
    mejor_penalizacion_global = float('inf')

    # 2. Bucle de Generaciones
    for gen in range(GENERACIONES):
        puntuaciones_y_individuos = [(ind, calcular_fitness(ind, datos)) for ind in poblacion]
        puntuaciones_y_individuos.sort(key=lambda x: x[1])

        mejor_individuo_actual, mejor_penalizacion_actual = puntuaciones_y_individuos[0]

        if mejor_penalizacion_actual < mejor_penalizacion_global:
            mejor_penalizacion_global = mejor_penalizacion_actual
            mejor_asignacion_global = copy.deepcopy(mejor_individuo_actual)

        print(f"Generación {gen + 1}/{GENERACIONES}: Mejor Penalización = {mejor_penalizacion_actual}")

        nueva_poblacion = [ind for ind, score in puntuaciones_y_individuos[:ELITISMO_COUNT]]

        while len(nueva_poblacion) < POBLACION_SIZE:
            padre1 = seleccionar_padre_por_torneo(puntuaciones_y_individuos)
            padre2 = seleccionar_padre_por_torneo(puntuaciones_y_individuos)

            if random.random() < TASA_CRUCE:
                hijo = cruzar(padre1, padre2, datos['escritorios_por_empleado'])
            else:
                hijo = copy.deepcopy(random.choice([padre1, padre2]))

            hijo_mutado = mutar(hijo, datos['employees'], datos['days'], datos['escritorios_por_empleado'])
            nueva_poblacion.append(hijo_mutado)

        poblacion = nueva_poblacion

    print("\n" + "="*50)
    print(f"Mejor Puntuación de Penalización Encontrada: {mejor_penalizacion_global}")
    print("--- Mejor Asignación Encontrada ---")
    for (e, d), desk in mejor_asignacion_global.items():
        print(f"  - Empleado: {e}, Día: {d} -> Escritorio: {desk}")
    print("="*50)

    # Generar y retornar KPIs
    kpis_finales = reportar_resultados(mejor_asignacion_global, datos) # This now returns a dict

    # ====================================================================
    # NUEVA SECCIÓN: IMPRIMIR KPIS DIRECTAMENTE EN LA SALIDA DE COLAB
    # ====================================================================
    print("\n" + "="*50)
    print("RESUMEN DE KPIs FINALES:")
    print("="*50)

    print("\n--- 1. KPIs de Cumplimiento (Restricciones Duras) ---")
    print(f"  • Empleados con asignación: {kpis_finales['empleados_con_asignacion_pct']:.2f}% (Ideal: 100%)")
    print(f"  • Escritorios con sobre-asignación: {kpis_finales['escritorios_con_sobre_asignacion']} (Ideal: 0)")
    print(f"  • Asignaciones en escritorios permitidos: {kpis_finales['asignaciones_en_escritorios_permitidos_pct']:.2f}% (Ideal: 100%)")

    print("\n--- 2. KPIs de Calidad (Restricciones Suaves) ---")
    print(f"  • Cumplimiento promedio de días preferidos: {kpis_finales['cumplimiento_promedio_dias_preferidos_pct']:.2f}% (Más alto es mejor)")
    print(f"  • Promedio de días asignados no-preferidos: {kpis_finales['promedio_dias_asignados_no_preferidos']:.2f} (Más bajo es mejor)")
    print(f"  • Empleados con escritorio único: {kpis_finales['empleados_con_escritorio_unico_pct']:.2f}% (Más alto es mejor)")

    print("\n--- 3. KPIs de Colaboración ---")
    print(f"  • Porcentaje de asignaciones aisladas: {kpis_finales['porcentaje_asignaciones_aisladas']:.2f}% (Más bajo es mejor)")
    print("  • Detalle de Reuniones de Grupo:")
    for item in kpis_finales['detalle_reuniones_grupo']:
        print(f"      - {item}")
    print(f"  • Grupos con reunión completa: {kpis_finales['grupos_con_reunion_completa_pct']:.2f}% (Más alto es mejor)")

    print("\n--- 4. KPIs de Eficiencia ---")
    print("  • Capacidad Utilizada por Día:")
    for day, info in kpis_finales['capacidad_utilizada_por_dia'].items():
        print(f"      - Día '{day}': {info['ocupados']}/{info['total_escritorios']} escritorios ({info['porcentaje']:.2f}%)")
    print(f"  • Porcentaje de Capacidad Utilizada GLOBAL: {kpis_finales['porcentaje_capacidad_utilizada_global']:.2f}% (Más alto es mejor)")

    print("\n" + "="*50)
    print("FIN DEL RESUMEN DE KPIs")
    print("="*50)
    # ====================================================================

    # Generar reportes finales
    generar_reporte_json(mejor_asignacion_global, datos, "reporte_asignaciones.json")
    generar_reporte_excel(mejor_asignacion_global, kpis_finales, "reporte_asignaciones.xlsx")

    graficos_paths = generar_graficos(kpis_finales)
    generar_reporte_pdf(mejor_penalizacion_global, mejor_asignacion_global, kpis_finales, graficos_paths, "reporte_final.pdf")

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":
    uploaded_file = None
    try:
        # Prompt the user to upload a JSON file
        print("Please upload your 'data.json' file:")
        uploaded = files.upload() # This opens the file selection dialog

        # Get the name of the uploaded file (assuming only one is uploaded)
        if uploaded:
            uploaded_file_name = list(uploaded.keys())[0]
            print(f"File '{uploaded_file_name}' uploaded successfully.")

            # Load the data from the uploaded file
            with open(uploaded_file_name, 'r') as f:
                datos_cargados = json.load(f)

            # Now, execute the genetic algorithm with the loaded data
            ejecutar_algoritmo_genetico(datos_cargados)
        else:
            print("No file was uploaded. Exiting.")

    except Exception as e:
        print(f"An error occurred during file upload or processing: {e}")
        print("Please ensure you upload a valid JSON file with the correct structure.")
