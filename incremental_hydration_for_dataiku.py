import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import numpy as np
import logging
from neo4j import GraphDatabase
import datetime
from typing import Dict
import warnings
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class DataLoaderAndMigrator:

    def __init__(self):

        #-- LOADER --#

        self.db_host = DatabaseConfig.host
        self.db_port = DatabaseConfig.port
        self.db_username = DatabaseConfig.username
        self.db_password = DatabaseConfig.password
        self.db_database = DatabaseConfig.database
        self.db_query1 = DatabaseConfig.query1
        self.db_query2 = DatabaseConfig.query2
        self.db_query3 = DatabaseConfig.query3
        self.db_schema = DatabaseConfig.schema
        self.last_sync_date_time = None
        self.time_sync()
        self.executor()

        #-- MIGRATION --#

        self.nj_url = Neo4jConfig.url
        self.nj_username = Neo4jConfig.username
        self.nj_password = Neo4jConfig.password

        self.driver = GraphDatabase.driver(self.nj_url, auth=(self.nj_username, self.nj_password))

    
    def time_sync(self):
        current_time = datetime.datetime.now()
        last_sync_date_time = current_time - datetime.timedelta(days=1)
        self.last_sync_date_time = pd.to_datetime(last_sync_date_time, format='mixed').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
    def executor(self):
        self.conn_string = self.database_connector(
            db_type='postgresql',
            host=self.db_host,
            port=self.db_port,
            database=self.db_database,
            username=self.db_username,
            password=self.db_password,
            schema=self.db_schema
            )
        
        logger.info("=" * 50)
        logger.info("LOADING DATA")
        logger.info("=" * 50)

        self.load_and_save_data()
        self.load_CSVs()
        self.data_preprocessor()
        self.save_neo4j_CSVs()
        self.create_and_save_relationships()

        logger.info("=" * 50)
        logger.info("DATA LOADING SUCCESSFUL!")
        logger.info("=" * 50)

    def database_connector(self, db_type, host, port, database, username, password, **kwargs):
    
        encoded_password = urllib.parse.quote_plus(password)
        connection_strings = {
            'postgresql': f"postgresql://{username}:{encoded_password}@{host}:{port}/{database}",
        }
        return connection_strings[db_type]
    
    def load_and_save_data(self):

        try:
            engine = create_engine(self.conn_string)
            
            if not self.db_query1:
                logger.warning("Query for v_request is missing!")
                return None 
            else:
                self.db_query1 = self.db_query1.replace(';',f'\nWHERE "requestModifiedDate" >=\'{self.last_sync_date_time}\';')
        
            if not self.db_query2:
                logger.warning("Query for v_assets is missing!")
                return None

            if not self.db_query3:
                logger.warning("Query for v_request_with_activities is missing!")
                return None
            else:
                self.db_query3 = self.db_query3.replace(';',f'\nWHERE "requestModifiedDate" >=\'{self.last_sync_date_time}\';')

            self.df_request = pd.read_sql(self.db_query1, engine)
            self.df_assets = pd.read_sql(self.db_query2, engine)
            # self.df_assets = assets_df #pd.read_csv('./fetched_data/v_assets.csv') 
            self.df_request_with_activities = pd.read_sql(self.db_query3, engine)
            
            logger.info(f" Downloaded {len(self.df_request)} rows from 'v_requests', {len(self.df_assets)} rows from 'v_assets', and {len(self.df_request_with_activities)} rows from 'v_request_with_activities'.")
            
        except Exception as e:
            logger.warning(f"Error connecting to database: {e}")
            return None
        
        finally:
            if 'engine' in locals():
                engine.dispose()
    
    def load_CSVs(self):
        try:
            self.is_hvac_df = HVAC_df
            self.suggested_asset_df = asset_suggest_df
            self.vendor_data = vendors_df

            logger.info("Helper CSV files loaded successfully.")

        except Exception as e:
            logger.warning(f"Error loading CSVs: {e}")
            

    def data_preprocessor(self):

        # Activity:
        activity_df = self.df_request_with_activities[self.df_request_with_activities['activityAlternateId'].notna()][['providertype','activityAlternateId','activityDescription']]
        activity_df.drop_duplicates(inplace = True)

        # Asset:
        requests_subset = self.df_request[['requestId','assetAlternateId', 'requestAlternateId']]
        requests_subset = requests_subset[requests_subset.assetAlternateId.notna()]

        v_assets = self.df_assets[['assetId','Asset Alt Id', 'Asset Description', 'manufacturer', 'model', 'serialNumber']]
        v_assets = v_assets.merge(requests_subset, left_on = 'Asset Alt Id', right_on = 'assetAlternateId', how= 'left')
        v_assets = v_assets[v_assets['requestAlternateId'].notna()] # keeping only those asset records which are associated to the presently fetched serviceRequests

        is_hvac_df = self.is_hvac_df.copy()
        is_hvac_df['is_HVAC'] = True
        is_hvac_df.drop(columns=['Asset Description'], inplace = True)
        v_assets_with_hvac = v_assets.merge(is_hvac_df, on='Asset Alt Id', how='left')
        
        if not v_assets_with_hvac.empty:
            v_assets_with_hvac.loc[v_assets_with_hvac['is_HVAC'] == True, 'asset_type'] = 'HVAC'
        else:
            v_assets_with_hvac['asset_type'] = None

        final_assets_df = v_assets_with_hvac[['assetId', 'Asset Description', 'Asset Alt Id', 'manufacturer', 'model',
                                              'serialNumber', 'is_HVAC', 'asset_type', 'requestId','assetAlternateId', 'requestAlternateId']]
        final_assets_df.loc[:, 'is_HVAC'] = final_assets_df['is_HVAC'].fillna(False)

        suggested_asset_df = self.suggested_asset_df.copy()
        suggested_asset_df.rename(columns={'asset_id': 'suggested_asset'}, inplace=True)
        suggested_asset_df_subset = suggested_asset_df[['request_id', 'suggested_asset']]

        final_assets_df = final_assets_df.merge(suggested_asset_df_subset, left_on = 'requestId', right_on = 'request_id', how = 'left')
        final_assets_df = final_assets_df[['assetId', 'Asset Description', 'Asset Alt Id', 'manufacturer', 'model',
                                           'serialNumber', 'is_HVAC', 'asset_type', 'suggested_asset','requestAlternateId']]
        
        vendor_data = self.vendor_data.copy()
        vendor_data = vendor_data[['requestAlternateId','vendorName', 'vendorAddress1', 'vendorCity',
                           'vendorRegion', 'vendorCountry', 'vendorPostalCode']]
        assets_with_vendors = final_assets_df.merge(vendor_data, on = 'requestAlternateId', how = 'left')
        assets_df = assets_with_vendors[['Asset Description', 'Asset Alt Id', 'manufacturer', 'model','serialNumber', 
                                                   'is_HVAC', 'asset_type', 'suggested_asset','vendorName', 'vendorAddress1', 'vendorCity',
                                                   'vendorRegion', 'vendorCountry', 'vendorPostalCode']]

        # Country:
        country_df = self.df_request[['country']].drop_duplicates()

        # Customer:
        customer_df = self.df_request[['customer']].drop_duplicates()

        # Location:
        location_df = self.df_request[['locationAlternateId', 'locationPath']].drop_duplicates()

        # Service Requests:
        temp_ser_req = self.df_request[['isSelfAssign', 'priorityCode', 
                  'requestCreatedDate', 'requestDescription', 'requestAlternateId', 'completionNotes', 
                  'requestTargetCompletionDate', 'serviceClassificationAlternateId', 'serviceClassificationPath',  
                  'requestCompletionDate', 'workType']]
        
        def to_local_datetime(date_col):
    
            if date_col is None:
                return None
            
            if date_col.isna().all():
                return date_col
            
            dt_series = pd.to_datetime(date_col, format='mixed')
            formatted = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
            
            return formatted.str.replace(' ', 'T')


        def process_service_requests(df_service_request):
                
                date_cols = ['requestCreatedDate', 'requestTargetCompletionDate', 'requestCompletionDate']
                
                for col in date_cols:
                    if col in df_service_request.columns:
                        # df_service_request.loc[:, col] = to_local_datetime(df_service_request[col]).astype(str) # not working- pandas replaces 'T' with a ' ' again
                        df_service_request[col] = to_local_datetime(df_service_request[col])
                
                df_service_request['createdYear'] = pd.to_datetime(df_service_request['requestCreatedDate']).dt.year
                df_service_request['createdMonth'] = pd.to_datetime(df_service_request['requestCreatedDate']).dt.month
                
                df_service_request['isCompleted'] = df_service_request['requestCompletionDate'].notna()   
                
                conditions = [
                    df_service_request['requestCompletionDate'].isna(),
                    df_service_request['requestTargetCompletionDate'].isna(),
                    df_service_request['requestCompletionDate'] <= df_service_request['requestTargetCompletionDate'],
                    df_service_request['requestCompletionDate'] > df_service_request['requestTargetCompletionDate']
                ]
                
                choices = ['Open', 'Open', 'Met', 'Miss']
                
                df_service_request['sla'] = np.select(conditions, choices, default='Unknown')
                
                return df_service_request

        
        service_req_df = process_service_requests(temp_ser_req)

        self.activity_df = activity_df.copy()
        self.assets_df = assets_df.copy()
        self.country_df = country_df.copy()
        self.customer_df = customer_df.copy()
        self.location_df = location_df.copy()
        self.service_req_df = service_req_df.copy()
        logger.info("Node and Property data prepared!")

    def save_neo4j_CSVs(self):
        
        try:
            # renaming the features:
            self.activity_df.rename(columns={'activityAlternateId': 'activityId', 'providertype':'providerType'}, inplace=True)

            self.assets_df.rename(columns={'Asset Alt Id': 'assetId', 'Asset Description':'assetDescription', 'vendorAddress1':'vendorAddress'}, inplace=True)

            self.location_df.rename(columns={'locationAlternateId': 'locationId'}, inplace=True)

            self.service_req_df.rename(columns={'requestAlternateId': 'requestId', 'serviceClassificationAlternateId': 'serviceClassificationId'}, inplace=True)

            # logger.info(f"Data for migration to Neo4J is saved on path: {neo4j_dir_path} and ready to be imported!")
        
        except Exception as e:
            logger.warning(f"Error Renaming Features: {e}")
    
    def create_and_save_relationships(self):

        try:

            self.LOCATED_AT = self.df_request[['assetAlternateId','locationAlternateId']].dropna().drop_duplicates()
            self.LOCATED_AT.rename(columns={'assetAlternateId': 'assetId', 'locationAlternateId': 'locationId'}, inplace=True)
            # self.LOCATED_AT.to_csv(f"{neo4j_relationship_dir_path}/LOCATED_AT.csv",index=False)

            self.AT_LOCATION = self.df_request[['requestAlternateId','locationAlternateId']].dropna().drop_duplicates()
            self.AT_LOCATION.rename(columns={'requestAlternateId': 'requestId', 'locationAlternateId': 'locationId'}, inplace=True)
            # self.AT_LOCATION.to_csv(f"{neo4j_relationship_dir_path}/AT_LOCATION.csv",index=False)

            self.HAS_ACTIVITY = self.df_request_with_activities[['activityAlternateId', 'requestAlternateId']].dropna().drop_duplicates()
            self.HAS_ACTIVITY.rename(columns={'requestAlternateId': 'requestId', 'activityAlternateId': 'activityId'}, inplace=True)
            # self.HAS_ACTIVITY.to_csv(f"{neo4j_relationship_dir_path}/HAS_ACTIVITY.csv",index=False)

            self.FOR_ASSET = self.df_request[['requestAlternateId','assetAlternateId']].dropna().drop_duplicates()
            self.FOR_ASSET.rename(columns={'requestAlternateId': 'requestId', 'assetAlternateId': 'assetId'}, inplace=True)
            # self.FOR_ASSET.to_csv(f"{neo4j_relationship_dir_path}/FOR_ASSET.csv",index=False)

            self.OPERATES_IN = self.df_request[['customer','country']].dropna().drop_duplicates()
            # self.OPERATES_IN.to_csv(f"{neo4j_relationship_dir_path}/OPERATES_IN.csv",index=False)

            self.RESIDES_AT = self.df_request[['customer','locationAlternateId']].dropna().drop_duplicates()
            self.RESIDES_AT.rename(columns={'locationAlternateId': 'locationId'}, inplace=True)
            # self.RESIDES_AT.to_csv(f"{neo4j_relationship_dir_path}/RESIDES_AT.csv",index=False)

            self.OWNS = self.df_request[['customer','assetAlternateId']].dropna().drop_duplicates()
            self.OWNS.rename(columns={'assetAlternateId': 'assetId'}, inplace=True)
            # self.OWNS.to_csv(f"{neo4j_relationship_dir_path}/OWNS.csv",index=False)

            self.CREATES = self.df_request[['customer','requestAlternateId']].dropna().drop_duplicates()
            self.CREATES.rename(columns={'requestAlternateId': 'requestId'}, inplace=True)
            # self.CREATES.to_csv(f"{neo4j_relationship_dir_path}/CREATES.csv",index=False)

            self.IN = self.df_request[['country','locationAlternateId']].dropna().drop_duplicates()
            self.IN.rename(columns={'locationAlternateId': 'locationId'}, inplace=True)
            # self.IN.to_csv(f"{neo4j_relationship_dir_path}/IN.csv",index=False)

            logger.info(f"Relationships created!")

        except Exception as e:

            logger.warning(f"Error while creating and saving relationships: {e}")


    def close(self):
        self.driver.close()
    
    def clear_database(self):

        with self.driver.session() as session:

            # Delete all nodes and relationships:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("All nodes and relationships deleted")
            
            # Delete all indexes:
            result = session.run("SHOW INDEXES")
            for record in result:
                index_name = record.get("name") or record.get("indexName")
                if index_name:
                    try:
                        session.run(f"DROP INDEX {index_name}")
                        logger.info(f"Dropped index: {index_name}")
                    except Exception as e:
                        logger.warning(f"Could not drop index {index_name}: {e}")
            
            # Delete all constraints:
            result = session.run("SHOW CONSTRAINTS")
            for record in result:
                constraint_name = record.get("name")
                if constraint_name:
                    try:
                        session.run(f"DROP CONSTRAINT {constraint_name}")
                        logger.info(f"Dropped constraint: {constraint_name}")
                    except Exception as e:
                        logger.warning(f"Could not drop constraint {constraint_name}: {e}")
            
            logger.info("Database completely cleared")


    def load_nodes_from_csv(self, csv, node_label: str, id_property: str, batch_size: int = 1000):
        """Load nodes from CSV file in batches."""
        df = csv
        df = df.where(pd.notnull(df), None)  # Replace NaN with None
        
        total_rows = len(df)
        # logger.info(f"Loading {total_rows} {node_label} nodes from {csv_path}")
        
        with self.driver.session() as session:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                records = batch.to_dict('records')
                
                # Build Cypher query dynamically
                query = f"""
                UNWIND $records AS record
                MERGE (n:{node_label} {{{id_property}: record.{id_property}}})
                SET n += record
                """
                session.run(query, records=records)
                logger.info(f"Loaded batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} for {node_label}")
        
        logger.info(f"Completed loading {node_label} nodes")


    def load_relationships_from_csv(self, csv, rel_config: Dict, batch_size: int = 1000):
        """
        Load relationships from CSV file.
        
        rel_config example:
        {
            'rel_type': 'LOCATED_AT',
            'from_label': 'Asset',
            'from_id_col': 'assetId',
            'from_id_prop': 'assetId',
            'to_label': 'Location',
            'to_id_col': 'locationId',
            'to_id_prop': 'locationId',
            'properties': []  # Optional: list of relationship properties
        }
        """
        df = csv
        df = df.where(pd.notnull(df), None)
        
        total_rows = len(df)
        # logger.info(f"Loading {total_rows} {rel_config['rel_type']} relationships from {csv_path}")
        
        with self.driver.session() as session:
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                records = batch.to_dict('records')
                
                # Build relationship properties string if any
                rel_props = ""
                if rel_config.get('properties'):
                    props_str = ", ".join([f"{p}: record.{p}" for p in rel_config['properties']])
                    rel_props = f" {{{props_str}}}"
                
                query = f"""
                UNWIND $records AS record
                MATCH (from:{rel_config['from_label']} {{{rel_config['from_id_prop']}: record.{rel_config['from_id_col']}}})
                MATCH (to:{rel_config['to_label']} {{{rel_config['to_id_prop']}: record.{rel_config['to_id_col']}}})
                MERGE (from)-[r:{rel_config['rel_type']}]->(to)
                """
                
                if rel_props:
                    query += f"\nSET r += {{{', '.join([f'{p}: record.{p}' for p in rel_config['properties']])}}}"
                
                session.run(query, records=records)
                logger.info(f"Loaded batch {i//batch_size + 1}/{(total_rows-1)//batch_size + 1} for {rel_config['rel_type']}")
        
        logger.info(f"Completed loading {rel_config['rel_type']} relationships")
    

    def verify_load(self):
        """Verify the data load by counting nodes and relationships."""
        with self.driver.session() as session:
            # Count nodes
            node_labels = ['Activity', 'Asset', 'Country', 'Customer', 'Location', 'ServiceRequest']
            for label in node_labels:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = result.single()['count']
                logger.info(f"{label} nodes: {count}")
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            for record in result:
                logger.info(f"{record['type']} relationships: {record['count']}")

    
    def migration_executor(self):

        # # CAUTION: Deleting the existing graph!!!
        # logger.info("=" * 50)
        # logger.info("DELETING THE EXISTING GRAPH!!")
        # logger.info("=" * 50) 
        # self.clear_database()

        logger.info("=" * 50)
        logger.info("LOADING NODES")
        logger.info("=" * 50)        
        self.load_nodes()

        logger.info("=" * 50)
        logger.info("LOADING RELATIONSHIPS")
        logger.info("=" * 50)
        self.load_relationships()

        logger.info("=" * 50)
        logger.info("VERIFYING NODE AND RELATIONSHP CREATION.")
        logger.info("=" * 50)
        self.verify_load()

        logger.info("=" * 50)
        logger.info("DATA MIGRATION SUCCESSFUL!")
        logger.info("=" * 50)

        self.close()

    def load_nodes(self):

        try:
            
            # Load nodes
            
            self.load_nodes_from_csv(
                self.activity_df, 
                "Activity", 
                "activityId"
            )
            
            self.load_nodes_from_csv(
                self.assets_df, 
                "Asset", 
                "assetId"
            )
            
            self.load_nodes_from_csv(
                self.country_df, 
                "Country", 
                "country"
            )
            
            self.load_nodes_from_csv(
                self.customer_df, 
                "Customer", 
                "customer"
            )
            
            self.load_nodes_from_csv(
                self.location_df, 
                "Location", 
                "locationId"
            )
            
            self.load_nodes_from_csv(
                self.service_req_df, 
                "ServiceRequest", 
                "requestId"
            )

        except Exception as e:
            logger.warning(f"Error creating Nodes in Neo4j: {e}")


    def load_relationships(self):
        
        try:
            # Load relationships
            
            # Asset -> Location
            self.load_relationships_from_csv(
                self.LOCATED_AT,
                {
                    'rel_type': 'LOCATED_AT',
                    'from_label': 'Asset',
                    'from_id_col': 'assetId',
                    'from_id_prop': 'assetId',
                    'to_label': 'Location',
                    'to_id_col': 'locationId',
                    'to_id_prop': 'locationId'
                }
            )
            
            # ServiceRequest -> Location
            self.load_relationships_from_csv(
                self.AT_LOCATION,
                {
                    'rel_type': 'AT_LOCATION',
                    'from_label': 'ServiceRequest',
                    'from_id_col': 'requestId',
                    'from_id_prop': 'requestId',
                    'to_label': 'Location',
                    'to_id_col': 'locationId',
                    'to_id_prop': 'locationId'
                }
            )
            
            # ServiceRequest -> Activity
            self.load_relationships_from_csv(
                self.HAS_ACTIVITY,
                {
                    'rel_type': 'HAS_ACTIVITY',
                    'from_label': 'ServiceRequest',
                    'from_id_col': 'requestId',
                    'from_id_prop': 'requestId',
                    'to_label': 'Activity',
                    'to_id_col': 'activityId',
                    'to_id_prop': 'activityId'
                }
            )
            
            # ServiceRequest -> Asset
            self.load_relationships_from_csv(
                self.FOR_ASSET,
                {
                    'rel_type': 'FOR_ASSET',
                    'from_label': 'ServiceRequest',
                    'from_id_col': 'requestId',
                    'from_id_prop': 'requestId',
                    'to_label': 'Asset',
                    'to_id_col': 'assetId',
                    'to_id_prop': 'assetId'
                }
            )
            
            # Customer -> Country
            self.load_relationships_from_csv(
                self.OPERATES_IN,
                {
                    'rel_type': 'OPERATES_IN',
                    'from_label': 'Customer',
                    'from_id_col': 'customer',
                    'from_id_prop': 'customer',
                    'to_label': 'Country',
                    'to_id_col': 'country',
                    'to_id_prop': 'country'
                }
            )
            
            # Customer -> Location
            self.load_relationships_from_csv(
                self.RESIDES_AT,
                {
                    'rel_type': 'RESIDES_AT',
                    'from_label': 'Customer',
                    'from_id_col': 'customer',
                    'from_id_prop': 'customer',
                    'to_label': 'Location',
                    'to_id_col': 'locationId',
                    'to_id_prop': 'locationId'
                }
            )
            
            # Customer -> Asset
            self.load_relationships_from_csv(
                self.OWNS,
                {
                    'rel_type': 'OWNS',
                    'from_label': 'Customer',
                    'from_id_col': 'customer',
                    'from_id_prop': 'customer',
                    'to_label': 'Asset',
                    'to_id_col': 'assetId',
                    'to_id_prop': 'assetId'
                }
            )
            
            # Customer -> ServiceRequest
            self.load_relationships_from_csv(
                self.CREATES,
                {
                    'rel_type': 'CREATES',
                    'from_label': 'Customer',
                    'from_id_col': 'customer',
                    'from_id_prop': 'customer',
                    'to_label': 'ServiceRequest',
                    'to_id_col': 'requestId',
                    'to_id_prop': 'requestId'
                }
            )
            
            # Location -> Country
            self.load_relationships_from_csv(
                self.IN,
                {
                    'rel_type': 'IN',
                    'from_label': 'Location',
                    'from_id_col': 'locationId',
                    'from_id_prop': 'locationId',
                    'to_label': 'Country',
                    'to_id_col': 'country',
                    'to_id_prop': 'country'
                }
            )

        except Exception as e:
            logger.warning(f"Error creating Relationships for nodes on Neo4j: {e}")


try:
    dataMigrator = DataLoaderAndMigrator()
    dataMigrator.migration_executor()
    
except NameError:
    logger.warning("The DataMigrator class is not defined. Please ensure it is defined correctly.")