"""Gera data/olist_entregas.csv.gz (um pedido entregue por linha) a partir dos CSVs originais da Olist.

Uso:  python scripts/preparar_olist.py caminho/para/csvs_da_olist
Dados: Brazilian E-Commerce Public Dataset by Olist (Kaggle, CC BY-NC-SA 4.0).
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATAS = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
         "order_delivered_customer_date", "order_estimated_delivery_date"]


def preparar(origem: Path) -> pd.DataFrame:
    pedidos = pd.read_csv(origem / "olist_orders_dataset.csv", parse_dates=DATAS)
    clientes = pd.read_csv(origem / "olist_customers_dataset.csv")
    itens = pd.read_csv(origem / "olist_order_items_dataset.csv")
    vendedores = pd.read_csv(origem / "olist_sellers_dataset.csv")

    # Estado do vendedor do primeiro item (99% dos pedidos têm um único vendedor)
    vendedor = itens.groupby("order_id").seller_id.first().map(vendedores.set_index("seller_id").seller_state)
    pedidos = pedidos[pedidos.order_status == "delivered"].merge(clientes[["customer_id", "customer_state"]], on="customer_id")
    tabela = pd.DataFrame({
        "pedido": pd.factorize(pedidos.order_id)[0],
        "compra": pedidos.order_purchase_timestamp,
        "aprovacao": pedidos.order_approved_at,
        "postagem": pedidos.order_delivered_carrier_date,   # entrega do pacote à transportadora
        "entrega": pedidos.order_delivered_customer_date,
        "prometido": pedidos.order_estimated_delivery_date,
        "uf_cliente": pedidos.customer_state,
        "uf_vendedor": pedidos.order_id.map(vendedor).values,
    })
    return tabela.dropna(subset=["compra", "postagem", "entrega"]).sort_values("compra")


if __name__ == "__main__":
    origem = Path(sys.argv[1])
    t = preparar(origem)
    t.to_csv(RAIZ / "data" / "olist_entregas.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"{len(t):,} pedidos")
