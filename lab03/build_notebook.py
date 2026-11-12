"""Build the executed lab notebook for criterion version 2."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parent


def build():
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md('''# Лабораторная № 3: The Well — acoustic scattering maze

**Метод № 4: семплирование с отклонением. Критерий версии 2.**

Цель — выбрать структуру акустического уравнения по точности, сложности и
устойчивости к интервенциям. Используются симуляции The Well и аналитический
benchmark CNO/RPB, 2D + время. Физических измерений нет.

Постановка взята из предоставленной первой лабораторной Ф. Соболевского
(`vendor/lab1_sobolevsky.ipynb`). Её Python-модули не предоставлялись;
реализация самостоятельная. Bootstrap-диагностика адаптирована из лабораторной № 2.

Исправление: ненулевой bootstrap-интервал больше не запрещает удалять ветвь.
В целевой функции учитывается прямой расчёт давления на валидации. Прежняя
версия сохранена в `results/criterion_v1/` и не используется для нового выбора.''')
    code('''from pathlib import Path
import sys, os, json
PROJECT=Path.cwd()
if PROJECT.name=='lab03': PROJECT=PROJECT.parent
sys.path.insert(0,str(PROJECT))
os.environ['MPLCONFIGDIR']=str(PROJECT/'tmp/lab03/mpl')
import numpy as np
from IPython.display import display, Image
from lab03.model import ROOT,TERMS,bits,graph
from lab03.analytic import archive_audit
from lab03.run import run

def figure(name):display(Image(filename=str(ROOT/'figures'/f'{name}.png')))
print('Criterion configuration:')
display(json.loads((ROOT/'criterion.json').read_text()))
''')
    md(r'''## 1. Исходное уравнение и граф

$$p_t=-K(u_x+v_y),\qquad u_t=-\rho^{-1}p_x,\qquad v_t=-\rho^{-1}p_y.$$

Для статических $K,\rho$ исключение скорости даёт
$$p_{tt}=K\partial_x(\rho^{-1}p_x)+K\partial_y(\rho^{-1}p_y).$$
Плотность задаёт лабиринт; выносить её за производную нельзя. Известны
$p(0)=p_0$, $p_t(0)=0$. Решение интегрируется из этих условий.

Расширенная библиотека:
$$p_{tt}=\theta_1F_x+\theta_2F_y+\theta_3F_{xy}
-\theta_4p_t/T_0-\theta_5p/T_0^2-\theta_6p^3/(P_0^2T_0^2).$$
Здесь $F_x=K\partial_x(\rho^{-1}p_x)$, $F_y=K\partial_y(\rho^{-1}p_y)$,
$F_{xy}=K[\partial_x(\rho^{-1}p_y)+\partial_y(\rho^{-1}p_x)]/2$,
$T_0=1$, $P_0=\max|p_0|$ фиксируется до интервенции.

Шесть переключателей дают 64 дерева выражений. Маска 3 — исходная структура,
63 — все члены. Коэффициенты подбираются только по обучению масштабированным
МНК/SVD. NetworkX хранит дерево и направленную матрицу смежности; одинаковые
операнды имеют отдельные вхождения, порядок аргументов сохраняется.

Динамическая структурная модель: $z_{n+1}=\Phi_G(z_n;\rho,K)+\epsilon_n$,
$z=(p,p_t)$. Временная развёртка ациклична. Физическое семейство задано заранее;
предпочтительность графа не доказывает причинность по произвольным наблюдениям.''')
    code('''import networkx as nx
g=graph(63)
print('Полный граф:',len(g),'вершин,',g.number_of_edges(),'рёбер')
print('DAG:',nx.is_directed_acyclic_graph(g))
''')
    md('''## 2. Данные и закрытая итоговая проверка

The Well, revision `8df383a3223f40f7ce66fe77b4ff4d7006dbc272`:
обучение — train/chunk_0/sample_0; валидация — valid/chunk_16/sample_0.
Берутся первые 101 кадр до t≈1.99, исходная сетка 256×256.
Фактические файлы имеют 202 кадра. Времена взяты из HDF5, физическая область
генератора — [-1,1]²; экспортные метки 0–1 не используются как физические центры.
Границы по метаданным и коду: нижние отражающие, верхние открытые на обеих осях.

**Прежняя test/sample_0 уже просматривалась и теперь считается разработочной.**
После фиксации новых весов и процедур в `criterion.json` для финальной оценки
используется **test/chunk_18/sample_1**, ранее не участвовавшая в выборе.

CNO: опубликованные случаи 0/1 — обучение/валидация. В исходном архиве они
обозначены OOD; наше разделение учебное. Прежний случай 2 сохраняется для
проверки точной формулы. Новый окончательный тест — независимые коэффициенты
с зафиксированным seed **55001**, K=32, r=0.85. Это наша реализация опубликованного
семейства, а не дополнительная архивная запись.''')
    code('''print('Published CNO endpoint audit:')
display(archive_audit())
for split,index in [('train',0),('valid',0)]:
    with np.load(ROOT/'data'/f'well_{split}_{index}.npz') as d:
        print(split,d['p'].shape,'time',float(d['t'][0]),float(d['t'][-1]))
''')
    md(r'''## 3. Аналитический benchmark и оценка моментов

$$p=\sum_{i,j=1}^{32}A_{ij}\sin(\pi ix)\sin(\pi jy)
\cos\left(0.1\pi t\sqrt{i^2+j^2}\right).$$
Коэффициенты опубликованных случаев восстановлены DST-I из поля на 64 узлах.
Поля сохраняют исходные амплитуды; min/max метаданные не применяются повторно.
Конечные архивные поля воспроизводятся до $3\cdot10^{-15}$.
Промежуточные кадры и производные — вычисления по формуле.

Для The Well используются приближённые слабые моменты: граневые потоки с
гармонической подвижностью, гауссово сглаживание (1.4 кадра, 2 ячейки),
производная временного ядра и исключение его краёв. Ошибка моментов
$E_m=\|X\hat\theta-y\|_2/\|y\|_2$ не равна ошибке давления.
На аналитике производные точные. Отдельно добавляется 1% коррелированного
шума только в обучающий $p_{tt}$: это вычислительное зашумление, не новая физика.''')
    md(r'''## 4. Bootstrap как диагностика, без запрета удаления

Сохраняется диагностика из лабораторной № 2: 1000 блочных реплик, блоки
4/8/12 полных кадров, 95%-е интервалы, повторный подбор при удалении ветви,
верхняя граница прироста ошибки моментов ≤0.01 и $W_1/SD_0\le1$ при условном
шуме остатков. Bootstrap-индексы базового и возмущённого опыта совпадают.

Пусть $b_j=1$, если прежнее консервативное правило советует сохранить ветвь.
Используем только ограниченный мягкий штраф
$$B(G)=\frac16\sum_{j\notin G}b_j\in[0,1].$$
Ненулевой интервал может отражать систематическую ошибку производных или
численную диссипацию. Поэтому он больше не обнуляет вероятность графа.
Интервалы условны на данной малой выборке; bootstrap не устраняет смещение.''')
    md(r'''## 5. Прямое решение на валидации и новый критерий

Все математически допустимые кандидаты решаются из начального поля:
The Well — Clawpack 5.9, unsplit MC, CFL 0.9; CNO — синусный Galerkin и DOP853.
Коэффициенты по-прежнему оцениваются только по обучению.

$$E_p(G)=\frac{\|p_G-p_{val}\|_2}{\|p_{val}\|_2},$$
$$J(G)=E_p(G)+0.1E_m(G)+0.02N(G)+0.5E_{int}(G)+0.01B(G).$$

$E_{int}$ — средняя **ошибка давления** при физических интервенциях:
амплитуда начального поля ×1.5, обмен осей со средой (The Well);
амплитуда ×1.5 и независимое начальное поле seed 3104 (CNO).
Для линейных кандидатов масштабирование рассчитывается по точной линейности;
для кубических выполняется новый расчёт с фиксированным $P_0$.
Преобразованные эталоны зависимы и не считаются новыми независимыми данными.

Жёсткие ограничения касаются пространственного тензора, возможности устойчиво
интегрировать модель и грубой ошибки давления **≤50% на валидации**.
Это не статистический запрет удаления. Для ускорения плохой кандидат можно
отклонить, когда уже накопленная сумма квадратов ошибок превышает 0.5² от
энергии всей эталонной траектории: будущие слагаемые неотрицательны.
Все веса и предел записаны до оценки новой тестовой траектории.

Кэш прямых расчётов содержит коэффициенты, ошибки, причины отказов и версию
солвера. Новый test/sample_1 и seed 55001 в этих расчётах не используются.''')
    md(r'''## 6. Семплирование с отклонением и корректность

В каждом раунде $q(G)=1/64$, независимо $U\sim\mathcal U(0,1)$.
Принимаем граф, если выполнены ограничения $I(G)=1$ и $U\le e^{-J(G)/T}$.
Поскольку $J\ge0$, ненормированная масса $q(G)I(G)e^{-J/T}\le q(G)$.
Следовательно,
$$P(G\mid accept)=\frac{q(G)I(G)e^{-J(G)/T}}
{\sum_Hq(H)I(H)e^{-J(H)/T}}.$$

Три раунда по 5000 предложений. Между раундами T уменьшается с учётом
квартиля принятых оценок; внутри раунда T и оценки фиксированы. Сохраняются
маски, U, принятия и причины отказов. Разные температуры не объединяются
для оценки одного стационарного распределения. Это мера качества, не posterior.

Малое семейство позволяет проверить распределение и минимум полным перебором.
Преимущество по скорости над перебором не заявляется. Семплирование за конечный
бюджет не гарантирует глобального оптимума вне этого семейства.''')
    code('''report=run(n_boot=1000)
from lab03.plotting import make_figures
make_figures()
print('Fresh test protocol:',report['evaluation'])
''')
    md('## 7. Исправление на одной валидационной траектории\n\nСравнение исходной структуры (3) и структуры с затуханием (11) использует одинаковые данные и Clawpack. Обе структуры теперь допустимы; bootstrap даёт только небольшой штраф.')
    code('''figure('criterion_comparison')
rows=json.loads((ROOT/'results/well_catalog.json').read_text())
for mask in (3,11):
    q=rows[mask]
    print('Mask',mask,'allowed:',q['gate'],'pressure validation:',q['validation_pressure'],
          'bootstrap risk:',q['bootstrap_risk'],'J:',q['J'])
''')
    md('## 8. Последовательности и проверка распределения')
    code('''figure('well_sequence')
figure('analytic_sequence')
figure('sampling_rounds')
figure('well_distribution')
figure('analytic_distribution')
for name in ('well','analytic','analytic_noise'):
    c=report[name]['comparison']
    print(name,'TV:',c['audit_TV'],'exhaustive best:',c['exhaustive_best_under_gate'])
''')
    md('## 9. Выбранные деревья и матрицы смежности')
    code('''figure('selected_branches')
for name in ('well','analytic','analytic_noise'):
    s=report[name]['selected']
    print(name,'mask:',s['mask'],'theta:',np.round(s['parameters'],8),
          'fresh test moment error:',s['test_moment_error'])
figure('well_tree')
figure('well_adjacency')
figure('analytic_tree')
''')
    md('## 10. Окончательная проверка на новых данных\n\nТест проводится после фиксации графа. Последующие кадры не подаются в солвер. Для The Well приводится Clawpack; независимая центральная схема остаётся проверкой чувствительности к дискретизации. Аналитическая задача интегрируется спектральным Galerkin + DOP853 и сравнивается с точной формулой.')
    code('''figure('well_data')
figure('well_approximation')
figure('well_error')
print('Fresh The Well pressure errors:')
display(report['well']['clawpack_original'])
display(report['well']['clawpack_selected'])
figure('analytic_approximation')
display(report['analytic']['selected_rollout'])
figure('convergence')
''')
    md('''## 11. Выводы и ограничения

Новая процедура сохраняет статистическую диагностику, но оценивает граф
по прямому решению на валидации. Исходная структура участвует в отборе;
bootstrap-интервал больше не превращается в запрет. Итоговый тест заменён,
поскольку прежняя траектория уже использовалась при анализе недостатка.

Вывод относится к заданной библиотеке и выбранным данным. По одной траектории
на часть недостаточно для общей оценки переноса между геометриями. Веса и
порог 50% — учебные решения. Интервенции частично зависимы; вычислительные
данные не являются физическими измерениями. Повторение seed семплера не
заменяет оценку неопределённости данных. Старые результаты не сравниваются
с новыми как единый тест — для сравнения критерия используется одна валидация.

## Источники и воспроизведение

- [The Well](https://polymathic-ai.org/the_well/datasets/acoustic_scattering_maze/),
  [данные и генератор](https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_maze).
- [CNO/RPB](https://zenodo.org/records/10406879),
  [Raonić et al., C.3.2](https://arxiv.org/html/2302.01178v2#A3.SS3.SSS2).
- [Goodman: rejection sampling](https://math.nyu.edu/~goodman/teaching/MonteCarlo07/notes/SimpleSampling.pdf).
- [Clawpack](https://www.clawpack.org/gallery/pyclaw/gallery/acoustics_2d_interface.html).
- [Лабораторная № 2](../lab02/README.md), исходная работа Ф. Соболевского в `vendor/`.

Инструкции, происхождение данных и ограничения: [README](README.md), `sources.json`.
''')
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (lab03)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    nb.write(notebook,ROOT/'lab03_acoustic_scattering.ipynb')
    print('Notebook built:',len(cells),'cells')


if __name__=='__main__':build()
