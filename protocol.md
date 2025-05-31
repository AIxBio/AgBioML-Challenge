# Bishop Lab Web Server Guide
Admin guide for setting up and maintaining the Bishop Lab web server. 

## Setting up the VM and Database
To start, you will need to set up a database and virtual machine (VM). Currently, we are using Azure -- which the university has a license for. We are running a web server on a B4ms machine (4 cores, 16G RAM, Ubuntu 18.04). Our MySQL server is on a Basic 1 Core with 500G of storage. 

To interact with these machines you will need the password for the account (username: `utadmin`).

## Setting up NGINX

Protocol for setting up NGINX on Ubuntu 18.04 and running multiple services

adapted from [here](https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04-quickstart).

1. Install NGINX

```
sudo apt update
sudo apt install nginx
```

2. Open firewall access to NGINX:

```
sudo ufw allow ssh  # Needed or you are locked out
sudo ufw allow 'Nginx HTTP'
sudo ufw enable
```

4. Connect to server on web browser, verify nginx is running

5. Make a directory structure, assign ownership to the <username> account:
  
```
sudo mkdir -p /var/www/bishop-laboratory/html
sudo chown -R $USER:$USER /var/www/bishop-laboratory/html
sudo chmod -R 755 /var/www/bishop-laboratory/
```

6. Test simple HTML page

```
vim /var/www/bishop-laboratory/html/index.html
```

add this:
  
```
<html>
    <head>
        <title>Welcome to Bishoplab.org!</title>
    </head>
    <body>
        <h1>Success!  The bishoplab.org server block is working!</h1>
    </body>
</html>
```

7. Make the corresponding server block (replace <ip address of server> with the actual ip address):

```
sudo vim /etc/nginx/sites-available/bishoplab
```

add:

```
server {

        # This is the port to listen to on the machine
        listen 80;
        listen [::]:80;

        # This is the root of the website's files
        root /var/www/bishop-laboratory/html;

        # This is the names for index files in that folder
        index index.html index.htm index.nginx-debian.html;

        # This is the address which traffic will arrive from
        server_name <ip address of server>;

        # This is the URL to serve these files at. '/' is the home.
        location / {
                try_files $uri $uri/ =404;
        }
}
```

8. Link server block to 'sites-enabled'

```
sudo ln -s /etc/nginx/sites-available/bishoplab /etc/nginx/sites-enabled/
```

9. Adjust hash bucket size setting in nginx.conf (remove comment character #)

```
sudo vim /etc/nginx/nginx.conf
```

add:

```
...
http {
    ...
    server_names_hash_bucket_size 64;
    ...
}
...
```

10. Test for errors:

```
sudo nginx -t
```

11. Reboot nginx

```
sudo systemctl restart nginx
```

12. View site in browser


## Serving a static website and a flask app using Nginx:

Let's say you want a flask app and a static site but you only have one server -- NGINX can do that.
https://stackoverflow.com/questions/11570321/configure-nginx-with-multiple-locations-with-different-root-folders-on-subdomain


optional:

1. Obtain a domain name 
2. Point the domain name to your server's IP address 

required:

3. Complete the steps from earlier to obtain a static web page
4. Complete these steps to obtain a flask app being run automatically through gunicorn:
https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04
5. Follow this guide to set it up so flask and a static page run simultaneously:
https://gist.github.com/soheilhy/8b94347ff8336d971ad0

final config file that worked for me:

```
server {
    listen 80;
    server_name gccri.bishop-lab.uthscsa.edu;
  
    # Increase upload size to 300MB (needed for shiny apps with large uploads)
    client_max_body_size 300M;

    # This is the URL to serve these files at. '/' is the home.
    location / {
        root /var/www/bishop-laboratory/html;
    }

    # Location for flask app
    location /<flask project name> {

        # Necessary to remove 'myproject' from the URL request
        # The URL request doesn't work unless its '/'
        rewrite ^/<flask project name>(.*) /$1 break;

        # Send traffic to the flask app
        include proxy_params;
        proxy_pass http://unix:/home/<flask author username>/<flask project name>/<flask project name>.sock;
    }
}
```

don't forget to run:

```
sudo nginx -t
sudo systemctl restart nginx
```

## Set up wordpress on this server

Since we're using Nginx, we're going to want to reverse proxy to the wordpress apache server:
https://www.digitalocean.com/community/tutorials/how-to-configure-nginx-as-a-web-server-and-reverse-proxy-for-apache-on-one-ubuntu-18-04-server


adapted from: https://tecadmin.net/install-wordpress-with-nginx-on-ubuntu/

1. Get PHP and apache:

```
sudo add-apt-repository ppa:ondrej/php
sudo apt-get update
sudo apt-get install php7.3 php7.3-fpm  mysql-server php7.3-mysql
```

2. Edit conf file to point to local port 9000

```
sudo vim/etc/php/7.3/fpm/pool.d/www.conf
```

comment this out:

```
;listen = /var/run/php5-fpm.sock
```

add this in:

```
listen = 127.0.0.1:9000
```

3. Get wordpress

```
wget http://wordpress.org/latest.tar.gz
tar xzf latest.tar.gz
sudo mv wordpress /var/www/bishoplab.org
sudo chown -R www-data.www-data /var/www/bishoplab.org
sudo chmod -R 755 /var/www/bishoplab.org
```

4. Create a db user for wordpress

```
sudo mysql 
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'root';
quit
```

5. Login as root and make the wordpress db

```
mysql -u root -p root
```

then:

```
CREATE DATABASE wp_db;
GRANT ALL ON wp_db.* to 'wp_user'@'localhost' IDENTIFIED BY <password for WP database user>;
FLUSH PRIVILEGES;
quit
```

5. Make the Nginx config file

```
sudo vim /etc/nginx/sites-available/bishoplab
```

add:

```
server {
    listen   80;

    root /var/www/bishoplab.org;
    index index.php index.html;
    server_name  gccri.bishop-lab.uthscsa.edu;
  
    # Increase upload size to 300MB (needed for shiny apps with large uploads)
    client_max_body_size 300M;

    location / {
            try_files $uri $uri/ /index.php?q=$request_uri;
    }

    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
          root /usr/share/nginx/www;
    }

    location ~ .php$ {
            try_files $uri =404;
            fastcgi_pass 127.0.0.1:9000;
            fastcgi_index index.php;
            fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
            include fastcgi_params;
    }

    # Location for flask app
    location /<flask project name> {

        # Necessary to remove 'myproject' from the URL request
        # The URL request doesn't work unless its '/'
        rewrite ^/<flask project name>(.*) /$1 break;

        # Send traffic to the flask app
        include proxy_params;
        proxy_pass http://unix:/home/<flask author username>/<flask project name>/<flask project name>.sock;
    }
}
```

6. Symbolic link and restart nginx (Remove any other enabled configs before this)

```
sudo ln -s /etc/nginx/sites-available/bishoplab /etc/nginx/sites-enabled
sudo nginx -t
sudo service nginx restart
sudo service php7.3-fpm restart
```

7. Go to your site in a browser. You should see the wordpress login. If this worked, congrats!! Best of luck if it didn't. I spent ~3 hours on this part.

8. Bonus: delete "powered by wordpress"

```
sudo vim /var/www/bishoplab.org/wp-content/themes/<your_theme>/footer.php
```

delete the line with 'powered by wordpress'

## Get SSL certs for nginx
Just go to https://certbot.eff.org/instructions

Make sure to enable https forwarding with your domain name service


## Migrate Word Press db to Azure/cloud MySQL

adapted from: https://geekflare.com/google-cloud-sql-wordpress/

1. Make a dump of current wordpress db

```
mysqldump -u root -p wp_db >/tmp/export.sql
```

2. Login to mysql in Azure and make wp database/user

```
mysql -h <host> -u <user> -p -P 3306
CREATE DATABASE wp_db;
CREATE USER wp_user@'%' IDENTIFIED BY '<wp db password>';
GRANT SELECT, INSERT, DELETE, UPDATE ON `wp_db`.* TO wp_user@'%';
```

## Add shiny apps to NGINX config
  
To add shiny apps, we use the directory method. This method, shown [here](https://support.rstudio.com/hc/en-us/articles/219002337-Shiny-Server-Quick-Start-Host-a-directory-of-applications), sets a directory of app directories for shinyserver to host rather than specifying them individually. 

1. make a shiny app directory

```
mkdir shinyapps/
sudo chmod -R 777 shinyapps/
```

2. Get shiny server

```
sudo apt-get install gdebi-core
wget https://download3.rstudio.org/ubuntu-18.04/x86_64/shiny-server-1.5.18.987-amd64.deb
sudo gdebi shiny-server-1.5.18.987-amd64.deb
```

3. Configure shiny server

```
sudo vim /etc/shiny-server/shiny-server.conf
```

add the following (replace `<username>`):

```
run_as www-data;

# Define a server that listens on port 3838
server {
  listen 3838;

  # For root shiny server (in shinyapps user home folder)
  location / {

    # Save logs here
    log_dir /var/log/shiny-server;

    # Path to shiny server for separate apps
    site_dir /home/<username>/shinyapps;

    # List contents of a (non-Shiny-App) directory when clients visit corresponding URIs
    directory_index on;
  
    # Set to 0 to prevent idleing -- not a great long-term solution though since it means memory usage is constant.
    app_idle_timeout 0;
  
  }

}
```

4. Configure Nginx to direct traffic to shiny-server (under http://url_of_site/shiny)

Add to NGINX config file:

```
location /shiny/ { 

        rewrite ^/shiny/(.*)$ /$1 break;
        proxy_pass http://localhost:3838;
        proxy_redirect http://localhost:3838/ $scheme://$host/shiny/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        #proxy_set_header Connection $connection_upgrade;
        proxy_read_timeout 20d;
        client_max_body_size 100m;

}
```

Full NGINX config now:

```


server {
        listen 443 default_server ssl;

        root /var/www/bishoplab.org;
        index index.php index.html;
        server_name  gccri.bishop-lab.uthscsa.edu;

        location / {
                try_files $uri $uri/ /index.php?q=$request_uri;
        }

        error_page 404 /404.html;
        error_page 500 502 503 504 /50x.html;
        location = /50x.html {
              root /usr/share/nginx/www;
        }

        location ~ .php$ {
                try_files $uri =404;
                fastcgi_pass 127.0.0.1:9000;
                fastcgi_index index.php;
                fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
                include fastcgi_params;
        }

        # Location for flask app
        location /myproject {

                # Necessary to remove 'myproject' from the URL request
                # The URL request doesn't work unless its '/'
                rewrite ^/myproject(.*) /$1 break;

                # Send traffic to the flask app
                include proxy_params;
                proxy_pass http://unix:/home/millerh1/myproject/myproject.sock;
        }

        location /shiny/ { # shiny server will locate at `http://example.domain/shiny/`

                rewrite ^/shiny/(.*)$ /$1 break;
                proxy_pass http://localhost:3838;
                proxy_redirect http://localhost:3838/ $scheme://$host/shiny/;
                proxy_http_version 1.1;
                proxy_set_header Upgrade $http_upgrade;
                #proxy_set_header Connection $connection_upgrade;
                proxy_read_timeout 20d;
                client_max_body_size 100m;

        }

        # Location for tomcat apps
        location /tomcat_apps/ {


                proxy_set_header X-Forwarded-Host $host;
                proxy_set_header X-Forwarded-Server $host;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

                # Send traffic to the tomcat server
                proxy_pass http://127.0.0.1:5000/;
        }


    ssl_certificate /etc/letsencrypt/live/gccri.bishop-lab.uthscsa.edu/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/gccri.bishop-lab.uthscsa.edu/privkey.pem; # managed by Certbot
}
server {
    if ($host = gccri.bishop-lab.uthscsa.edu) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


        listen   80;
        server_name  gccri.bishop-lab.uthscsa.edu;
    return 404; # managed by Certbot


}

```

### Stand up correlation analyzeR

1. Get the `correlationAnalyzeR` app, add libraries, and modify `tmp` permissions: 

```shell
git clone https://github.com/millerh1/correlationAnalyzeR-ShinyApp.git shinyapps/correlation-analyzer
sudo chown -R shiny:shiny shinyapps/correlation-analyzer/
sudo chmod -R 777 shinyapps/correlation-analyzer/www/tmp
cd shinyapps/correlation-analyzer
```

2. Install R v4.2.0+ and get the `renv` package

```R
R -e 'if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")'
```

3. Install `getSysReqs`

```shell
R -e 'if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")'
R -e 'remotes::install_github("mdneuzerling/getsysreqs", force=TRUE)'
```

4. Install system reqs

```shell
REQS=$(Rscript -e 'options(warn = -1); cat(getsysreqs::get_sysreqs("renv.lock"))' | sed s/"WARNING: ignoring environment value of R_HOME"//) \
  && echo $REQS \
  && sudo apt-get install -y $REQS
```

5. Optional: You may need `cmake > v3.15` to build `nlopt`. If so, follow these instructions to get the latest `cmake`: [link](https://askubuntu.com/a/865294/952008)

6. Create a password for the `shiny` user account and log into it

```shell
sudo passwd shiny
su - shiny
```

7. As the `shiny` account, restore the R environment to build all dependencies (this needs to be done as `shiny` so that the deps will be available to the shiny server)

```shell
R -e "renv::restore()"
```

8. Switch back to your main account and restart shiny-server

```
su - utadmin
sudo systemctl restart shiny-server
```

Your app should now be available at https://gccri.bishop-lab.uthscsa.edu/shiny/correlation-analyzer/
  
### Stand up RLBase

1. Get the `RLBase` app

```
git clone https://github.com/Bishop-Laboratory/RLBase.git shinyapps/rlbase
mkdir /var/www/.cache
chmod -R 777 /var/www/.cache # Necessary for access to the AnnotationHub and ExperimentHub cache
mkdir shinyapps/rlbase/app_cache
sudo chown -R shiny:shiny shinyapps/rlbase/
sudo chmod 777 -R shinyapps/rlbase/app_cache
cd shinyapps/rlbase/
```
  
2. Add AWS credentials needed for RLBase functionality in a file called `.Renviron` within the `shinyapps/rlbase` dir. Get these values from talking to Henry Miller (ask Dr. Bishop for Henry's contact info if he is no longer around).

```config
AWS_ACCESS_KEY_ID="<access_key>"
AWS_SECRET_ACCESS_KEY="<secret_access_key>"
AWS_DEFAULT_REGION="us-east-1"
```

4. Install system reqs

```shell
REQS=$(Rscript -e 'options(warn = -1); cat(getsysreqs::get_sysreqs("renv.lock"))' | sed s/"WARNING: ignoring environment value of R_HOME"//) \
  && echo $REQS \
  && sudo apt-get install -y $REQS
```

5. Restore the R environment to build all dependencies as `shiny`

```shell
su - shiny
R -e "renv::restore()"
```

6. Restart shiny-server as `utadmin`

```
su - utadmin
sudo systemctl restart shiny-server
```

RLBase should now be fully operational at this link: https://gccri.bishop-lab.uthscsa.edu/shiny/rlbase/

### Crontab to deal with zombie processes

Zombie processes from shiny can cause serious memory issues that eventually break the apps. The R-Shiny Server needs to be restarted at periodic intervals, and then the apps need to be reloaded.

The following script accomplishes that:

```shell
vim restart-server.sh
```

Includes this content:

```shell
#!/bin/bash

echo "Killing all R processes"
sudo killall -9 R

echo "Restarting shiny now"
sudo systemctl restart shiny-server.service

sleep 8
echo "Reloading apps"
curl https://gccri.bishop-lab.uthscsa.edu/shiny/rlbase/ &
curl https://gccri.bishop-lab.uthscsa.edu/shiny/correlation-analyzer/
```

Here is the crontab job:

```shell
sudo crontab -e
```

Content of job:

```crontab
0 0 * * * /home/utadmin/restart-server.sh >> /home/utadmin/cronlog.txt
```

Once this is set up, the cron job will run every day at midnight -- restarting the shiny server and reloading each app.