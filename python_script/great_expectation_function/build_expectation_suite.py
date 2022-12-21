import datetime
import pytz
import os
import json
from ruamel import yaml
import ruamel
from collections import defaultdict
import re
import pandas as pd
import numpy as np
import great_expectations 
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint.checkpoint import SimpleCheckpoint
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from great_expectations.profile.user_configurable_profiler  import UserConfigurableProfiler
import IPython
import great_expectations as ge
from typing import TypeVar
PandasDataFrame = TypeVar('pandas.core.frame.DataFrame')

class DataQuality():
    
    def __init__(self, root_directory: str, datasource_name:str , dataframe:PandasDataFrame  , partition_date: datetime.date):
        
        '''
        create great expectations context and default runtime datasource
        '''
        
        self.root_directory = root_directory
        self.datasource_name = datasource_name
        self.expectation_suite_name = f"{datasource_name}_expectation_suite"
        self.checkpoint_name = f"{datasource_name}_checkpoint"
        self.dataframe = dataframe
        self.partition_date = partition_date
        
        # create data context
        data_context_config = DataContextConfig( store_backend_defaults=FilesystemStoreBackendDefaults(root_directory=root_directory) )
        context = BaseDataContext(project_config = data_context_config)
     
        datasource_yaml = rf"""
        name: {self.datasource_name}
        class_name: Datasource
        module_name: great_expectations.datasource
        execution_engine:
          module_name: great_expectations.execution_engine
          class_name: PandasExecutionEngine
        data_connectors:
            default_runtime_data_connector_name:
                class_name: RuntimeDataConnector
                batch_identifiers:
                    - default_identifier_name
        """
        context.test_yaml_config(datasource_yaml)
        context.add_datasource(**yaml.load(datasource_yaml, Loader=ruamel.yaml.Loader))
        
        self.context = context
        file1 = open('/code/notebook/script/great_expectation_function/myfile.txt', 'w')
        file1.writelines([self.datasource_name])
    
    def get_context(self):
        
        '''
        retrieving data context in case you would like to manually extract / tweak the context by yourself
        '''
        
        return self.context
    
    def get_expectation_suit(self):
        
        '''
        retriving the current expectation suite
        '''
        
        return self.context.get_expectation_suite(self.expectation_suite_name)
    
    def create_batch_data(self, df, partition_date: datetime.date):
        
        '''
        create runtime batch request from the input dataframe and partition date
        '''
        
        batch_request = RuntimeBatchRequest(
            datasource_name= self.datasource_name,
            data_connector_name= "default_runtime_data_connector_name",
            data_asset_name=f"{self.datasource_name}_{self.partition_date.strftime('%Y%m%d')}",
            batch_identifiers={"default_identifier_name": "default_identifier"},
            runtime_parameters={"batch_data": df}
        )
        
        return batch_request
    
    def create_expectation_suite_if_not_exist(self):
        
        '''
        create expectation suite if not exist
        '''
        
        try:
            # create expectation suite
            self.context.create_expectation_suite(
                expectation_suite_name = self.expectation_suite_name,
                overwrite_existing=False
            )
        except great_expectations.exceptions.DataContextError as e:
            print(e)
        except Error as e:
            raise e
            
    def delete_expectation_suite(self):
        
        '''
        delete the expectation suite
        '''
        
        self.context.delete_expectation_suite(expectation_suite_name = self.expectation_suite_name)
   
    def delete_existing_expectation_suite(self):
        batch_request = self.create_batch_data(self.dataframe, self.partition_date)          
        # delete and create a new expectation suite
        try:
            self.delete_expectation_suite(expectation_suite_name)
        except:
            pass

        self.context.create_expectation_suite(
            expectation_suite_name = self.expectation_suite_name,
            overwrite_existing=True
        )

        validator = self.context.get_validator(
            batch_request = batch_request,
            expectation_suite_name = self.expectation_suite_name,
        )

        return validator
    
    def get_validator(self, with_profiled=False, append_suit = False):
        if not append_suit:
            '''
            retreiving a validator object for a fine grain adjustment on the expectation suite. and you can add some assert on this
            such as hight must be more than 0
            '''

            batch_request = self.create_batch_data(self.dataframe, self.partition_date)
            self.create_expectation_suite_if_not_exist()

            validator = self.context.get_validator(
                batch_request = batch_request,
                expectation_suite_name = self.expectation_suite_name,
            )
            
            if with_profiled:

                # build expectation with profiler
                not_null_only = True
                table_expectations_only = False

                profiler = UserConfigurableProfiler(
                    profile_dataset = validator,
                    not_null_only = not_null_only,
                    table_expectations_only = table_expectations_only
                )

                suite = profiler.build_suite()

                # save validation
                validator.save_expectation_suite(discard_failed_expectations=False)
        else:
            validator = self.delete_existing_expectation_suite()
            # domain knowledge
            # validator.expect_column_values_to_be_between('fixed acidity', min_value = 0, max_value = 14)
            # save your edited expectation suite to context
            validator.save_expectation_suite(discard_failed_expectations = False)
            # build expectation with profiler
            not_null_only = True
            table_expectations_only = False

            profiler = UserConfigurableProfiler(
                profile_dataset = validator,
                not_null_only = not_null_only,
                table_expectations_only = table_expectations_only
            )

            suite = profiler.build_suite()

            # save validation
            validator.save_expectation_suite(discard_failed_expectations=False)
            
            
        return validator
    
    def create_checkpoint_if_not_exist(self):
        
        '''
        create checkpoint if not exist.
        '''
        
        try:
            self.context.get_checkpoint(self.checkpoint_name)
            print(f'{self.checkpoint_name} is already created')
                    
        except great_expectations.exceptions.CheckpointNotFoundError:
            
            checkpoint_config = {
                "name": self.checkpoint_name,
                "config_version": 1.0,
                "class_name": "SimpleCheckpoint",
                "run_name_template": "%Y%m%d-%H%M%S",
            }
            self.context.test_yaml_config(yaml.dump(checkpoint_config))
            self.context.add_checkpoint(**checkpoint_config)

        except Error as e:
            raise e
            
            
    def validate_data(self, df=None, partition_date: datetime.date=None):
        
        '''
        validate dataset using the input dataset when initiated the class
        or user provided dataset when calling the method.
        '''
        
        if df and partition_date:
            batch_request = self.create_batch_data(df, partition_date)
        else:
            batch_request = self.create_batch_data(self.dataframe, self.partition_date)            
        
        self.create_checkpoint_if_not_exist()
        
        # run expectation_suite against data
        checkpoint_result = self.context.run_checkpoint(
            checkpoint_name = self.checkpoint_name,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": self.expectation_suite_name,
                }
            ],
        )
        
        for k,v in checkpoint_result['run_results'].items():
            self.render_file = v['actions_results']['update_data_docs']['local_site'].replace('file://', '')
    
        return checkpoint_result   